use arrow::array::{
    Array, BinaryViewArray, DictionaryArray, DictionaryKey, PrimitiveArray, Splitable,
    Utf8ViewArray,
};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::buffer::Buffer;
use arrow::datatypes::{ArrowDataType, IntegerType, PhysicalType};
use arrow::legacy::utils::CustomIterTools;
use arrow::trusted_len::TrustMyLength;
use arrow::types::NativeType;
use polars_compute::min_max::MinMaxKernel;
use polars_error::{PolarsResult, polars_bail};

use super::binary::{
    build_statistics as binary_build_statistics, encode_plain as binary_encode_plain,
};
use super::fixed_size_binary::{
    build_statistics as fixed_binary_build_statistics, encode_plain as fixed_binary_encode_plain,
};
use super::pages::PrimitiveNested;
use super::primitive::{
    build_statistics as primitive_build_statistics, encode_plain as primitive_encode_plain,
};
use super::{
    EncodeNullability, Nested, WriteOptions, binview, estimated_byte_size_to_values_per_page,
    nested,
};
use crate::arrow::read::schema::is_nullable;
use crate::arrow::write::{slice_nested_leaf, utils};
use crate::parquet::CowBuffer;
use crate::parquet::encoding::Encoding;
use crate::parquet::encoding::hybrid_rle::encode;
use crate::parquet::page::{DictPage, Page};
use crate::parquet::schema::types::PrimitiveType;
use crate::parquet::statistics::ParquetStatistics;
use crate::write::DynIter;

trait MinMaxThreshold {
    const DELTA_THRESHOLD: usize;
    const BITMASK_THRESHOLD: usize;

    fn from_start_and_offset(start: Self, offset: usize) -> Self;
}

macro_rules! minmaxthreshold_impls {
    ($($signed:ty, $unsigned:ty => $threshold:literal, $bm_threshold:expr,)+) => {
        $(
        impl MinMaxThreshold for $signed {
            const DELTA_THRESHOLD: usize = $threshold;
            const BITMASK_THRESHOLD: usize = $bm_threshold;

            fn from_start_and_offset(start: Self, offset: usize) -> Self {
                start + ((offset as $unsigned) as $signed)
            }
        }
        impl MinMaxThreshold for $unsigned {
            const DELTA_THRESHOLD: usize = $threshold;
            const BITMASK_THRESHOLD: usize = $bm_threshold;

            fn from_start_and_offset(start: Self, offset: usize) -> Self {
                start + (offset as $unsigned)
            }
        }
        )+
    };
}

minmaxthreshold_impls! {
    i8, u8 => 16, u8::MAX as usize,
    i16, u16 => 256, u16::MAX as usize,
    i32, u32 => 512, u16::MAX as usize,
    i64, u64 => 2048, u16::MAX as usize,
}

enum DictionaryDecision {
    NotWorth,
    TryAgain,
    Found(DictionaryArray<u32>),
}

fn min_max_integer_encode_as_dictionary_optional<'a, E, T>(
    array: &'a dyn Array,
) -> DictionaryDecision
where
    E: std::fmt::Debug,
    T: NativeType
        + MinMaxThreshold
        + std::cmp::Ord
        + TryInto<u32, Error = E>
        + std::ops::Sub<T, Output = T>
        + num_traits::CheckedSub
        + num_traits::cast::AsPrimitive<usize>,
    std::ops::RangeInclusive<T>: Iterator<Item = T>,
    PrimitiveArray<T>: MinMaxKernel<Scalar<'a> = T>,
{
    let min_max = <PrimitiveArray<T> as MinMaxKernel>::min_max_ignore_nan_kernel(
        array.as_any().downcast_ref().unwrap(),
    );

    let Some((min, max)) = min_max else {
        return DictionaryDecision::TryAgain;
    };

    debug_assert!(max >= min, "{max} >= {min}");
    let Some(diff) = max.checked_sub(&min) else {
        return DictionaryDecision::TryAgain;
    };

    let diff = diff.as_();

    if diff > T::BITMASK_THRESHOLD {
        return DictionaryDecision::TryAgain;
    }

    let mut seen_mask = MutableBitmap::from_len_zeroed(diff + 1);

    let array = array.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();

    if array.has_nulls() {
        for v in array.non_null_values_iter() {
            let offset = (v - min).as_();
            debug_assert!(offset <= diff);

            unsafe {
                seen_mask.set_unchecked(offset, true);
            }
        }
    } else {
        for v in array.values_iter() {
            let offset = (*v - min).as_();
            debug_assert!(offset <= diff);

            unsafe {
                seen_mask.set_unchecked(offset, true);
            }
        }
    }

    let cardinality = seen_mask.set_bits();

    let mut is_worth_it = false;

    is_worth_it |= cardinality <= T::DELTA_THRESHOLD;
    is_worth_it |= (cardinality as f64) / (array.len() as f64) < 0.75;

    if !is_worth_it {
        return DictionaryDecision::NotWorth;
    }

    let seen_mask = seen_mask.freeze();

    // SAFETY: We just did the calculation for this.
    let indexes = seen_mask
        .true_idx_iter()
        .map(|idx| T::from_start_and_offset(min, idx));
    let indexes = unsafe { TrustMyLength::new(indexes, cardinality) };
    let indexes = indexes.collect_trusted::<Vec<_>>();

    let mut lookup = vec![0u16; diff + 1];

    for (i, &idx) in indexes.iter().enumerate() {
        lookup[(idx - min).as_()] = i as u16;
    }

    use ArrowDataType as DT;
    let values = PrimitiveArray::new(DT::from(T::PRIMITIVE), indexes.into(), None);
    let values = Box::new(values);

    let keys: Buffer<u32> = array
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .unwrap()
        .values()
        .iter()
        .map(|v| {
            // @NOTE:
            // Since the values might contain nulls which have a undefined value. We just
            // clamp the values to between the min and max value. This way, they will still
            // be valid dictionary keys.
            let idx = *v.clamp(&min, &max) - min;
            let value = unsafe { lookup.get_unchecked(idx.as_()) };
            (*value).into()
        })
        .collect();

    let keys = PrimitiveArray::new(DT::UInt32, keys, array.validity().cloned());
    DictionaryDecision::Found(
        DictionaryArray::<u32>::try_new(
            ArrowDataType::Dictionary(
                IntegerType::UInt32,
                Box::new(DT::from(T::PRIMITIVE)),
                false, // @TODO: This might be able to be set to true?
            ),
            keys,
            values,
        )
        .unwrap(),
    )
}

pub(crate) fn encode_as_dictionary_optional(
    array: &dyn Array,
    nested: &[Nested],
    type_: PrimitiveType,
    options: WriteOptions,
) -> Option<PolarsResult<DynIter<'static, PolarsResult<Page>>>> {
    if array.is_empty() {
        let array = DictionaryArray::<u32>::new_empty(ArrowDataType::Dictionary(
            IntegerType::UInt32,
            Box::new(array.dtype().clone()),
            false, // @TODO: This might be able to be set to true?
        ));

        return Some(array_to_pages(
            &array,
            type_,
            nested,
            options,
            Encoding::RleDictionary,
        ));
    }

    use arrow::types::PrimitiveType as PT;
    let fast_dictionary = match array.dtype().to_physical_type() {
        PhysicalType::Primitive(pt) => match pt {
            PT::Int8 => min_max_integer_encode_as_dictionary_optional::<_, i8>(array),
            PT::Int16 => min_max_integer_encode_as_dictionary_optional::<_, i16>(array),
            PT::Int32 => min_max_integer_encode_as_dictionary_optional::<_, i32>(array),
            PT::Int64 => min_max_integer_encode_as_dictionary_optional::<_, i64>(array),
            PT::UInt8 => min_max_integer_encode_as_dictionary_optional::<_, u8>(array),
            PT::UInt16 => min_max_integer_encode_as_dictionary_optional::<_, u16>(array),
            PT::UInt32 => min_max_integer_encode_as_dictionary_optional::<_, u32>(array),
            PT::UInt64 => min_max_integer_encode_as_dictionary_optional::<_, u64>(array),
            _ => DictionaryDecision::TryAgain,
        },
        _ => DictionaryDecision::TryAgain,
    };

    match fast_dictionary {
        DictionaryDecision::NotWorth => return None,
        DictionaryDecision::Found(dictionary_array) => {
            return Some(array_to_pages(
                &dictionary_array,
                type_,
                nested,
                options,
                Encoding::RleDictionary,
            ));
        },
        DictionaryDecision::TryAgain => {},
    }

    let dtype = Box::new(array.dtype().clone());

    let estimated_cardinality = polars_compute::cardinality::estimate_cardinality(array);

    if array.len() > 128 && (estimated_cardinality as f64) / (array.len() as f64) > 0.75 {
        return None;
    }

    // This does the group by.
    let array = polars_compute::cast::cast(
        array,
        &ArrowDataType::Dictionary(IntegerType::UInt32, dtype, false),
        Default::default(),
    )
    .ok()?;

    let array = array
        .as_any()
        .downcast_ref::<DictionaryArray<u32>>()
        .unwrap();

    Some(array_to_pages(
        array,
        type_,
        nested,
        options,
        Encoding::RleDictionary,
    ))
}

fn serialize_def_levels_simple(
    validity: Option<&Bitmap>,
    length: usize,
    is_optional: bool,
    options: WriteOptions,
    buffer: &mut Vec<u8>,
) -> PolarsResult<()> {
    utils::write_def_levels(buffer, is_optional, validity, length, options.version)
}

fn serialize_keys_values<K: DictionaryKey>(
    array: &DictionaryArray<K>,
    validity: Option<&Bitmap>,
    buffer: &mut Vec<u8>,
) -> PolarsResult<()> {
    let keys = array.keys_values_iter().map(|x| x as u32);
    if let Some(validity) = validity {
        // discard indices whose values are null.
        let keys = keys
            .zip(validity.iter())
            .filter(|&(_key, is_valid)| is_valid)
            .map(|(key, _is_valid)| key);
        let num_bits = utils::get_bit_width(keys.clone().max().unwrap_or(0) as u64);

        let keys = utils::ExactSizedIter::new(keys, array.len() - validity.unset_bits());

        // num_bits as a single byte
        buffer.push(num_bits as u8);

        // followed by the encoded indices.
        Ok(encode::<u32, _, _>(buffer, keys, num_bits)?)
    } else {
        let num_bits = utils::get_bit_width(keys.clone().max().unwrap_or(0) as u64);

        // num_bits as a single byte
        buffer.push(num_bits as u8);

        // followed by the encoded indices.
        Ok(encode::<u32, _, _>(buffer, keys, num_bits)?)
    }
}

fn serialize_levels(
    validity: Option<&Bitmap>,
    length: usize,
    type_: &PrimitiveType,
    nested: &[Nested],
    options: WriteOptions,
    buffer: &mut Vec<u8>,
) -> PolarsResult<(usize, usize)> {
    if nested.len() == 1 {
        let is_optional = is_nullable(&type_.field_info);
        serialize_def_levels_simple(validity, length, is_optional, options, buffer)?;
        let definition_levels_byte_length = buffer.len();
        Ok((0, definition_levels_byte_length))
    } else {
        nested::write_rep_and_def(options.version, nested, buffer)
    }
}

fn normalized_validity<K: DictionaryKey>(array: &DictionaryArray<K>) -> Option<Bitmap> {
    match (array.keys().validity(), array.values().validity()) {
        (None, None) => None,
        (keys, None) => keys.cloned(),
        // The values can have a different length than the keys
        (_, Some(_values)) => {
            let iter = (0..array.len()).map(|i| unsafe { !array.is_null_unchecked(i) });
            MutableBitmap::from_trusted_len_iter(iter).into()
        },
    }
}

/// Serialize dictionary keys for flat (non-nested) arrays into multiple pages.
///
/// Page-level statistics are not currently supported because computing min/max
/// of dictionary values per page would require additional lookups. Column-level
/// statistics are still written via the dictionary page.
fn serialize_keys_flat<K: DictionaryKey>(
    array: &DictionaryArray<K>,
    type_: PrimitiveType,
    _statistics: Option<ParquetStatistics>,
    options: WriteOptions,
) -> PolarsResult<Vec<Page>> {
    // Parquet only accepts a single validity - we "&" the validities into a single one
    // and ignore keys whose _value_ is null.
    // It's important that we slice before normalizing.
    let validity = normalized_validity(array);
    let mut array = array.clone();
    array.set_validity(validity);

    // Early return for empty arrays to avoid division by zero in page estimation
    if array.is_empty() {
        return Ok(vec![]);
    }

    let estimated_bits_per_value = array.values().len().next_power_of_two().trailing_zeros() + 1;
    let estimated_bits_per_value = estimated_bits_per_value as usize;
    let estimated_byte_size = (estimated_bits_per_value * array.len()).div_ceil(8);

    let rows_per_page = estimated_byte_size_to_values_per_page(
        array.len(),
        estimated_byte_size,
        options.data_page_size,
    );

    let num_pages = array.len().div_ceil(rows_per_page);
    let mut data_pages = Vec::with_capacity(num_pages);

    while !array.is_empty() {
        let num_page_rows = rows_per_page.min(array.len());

        let page_array;
        (page_array, array) = array.split_at(num_page_rows);

        let mut buffer = vec![];

        let is_optional = is_nullable(&type_.field_info);
        serialize_def_levels_simple(
            page_array.validity(),
            num_page_rows,
            is_optional,
            options,
            &mut buffer,
        )?;
        let definition_levels_byte_length = buffer.len();

        serialize_keys_values(&page_array, page_array.validity(), &mut buffer)?;

        let page = utils::build_plain_page(
            buffer,
            num_page_rows, // num_values == num_rows when flat
            num_page_rows,
            page_array.null_count(),
            0, // flat means no repetition values
            definition_levels_byte_length,
            None, // we don't support writing page level statistics atm
            type_.clone(),
            options,
            Encoding::RleDictionary,
        )?;
        data_pages.push(Page::Data(page));
    }

    Ok(data_pages)
}

fn serialize_keys_nested<K: DictionaryKey>(
    array: &DictionaryArray<K>,
    type_: PrimitiveType,
    nested: &[Nested],
    statistics: Option<ParquetStatistics>,
    options: WriteOptions,
) -> PolarsResult<Vec<Page>> {
    let mut buffer = vec![];

    let (start, len) = slice_nested_leaf(nested);

    let mut nested = nested.to_vec();
    let array = array.clone().sliced(start, len);
    if let Some(Nested::Primitive(PrimitiveNested { length, .. })) = nested.last_mut() {
        *length = len;
    } else {
        unreachable!("")
    }
    // Parquet only accepts a single validity - we "&" the validities into a single one
    // and ignore keys whose _value_ is null.
    // It's important that we slice before normalizing.
    let validity = normalized_validity(&array);

    let (repetition_levels_byte_length, definition_levels_byte_length) = serialize_levels(
        validity.as_ref(),
        array.len(),
        &type_,
        &nested,
        options,
        &mut buffer,
    )?;

    serialize_keys_values(&array, validity.as_ref(), &mut buffer)?;

    let num_values = array.len();
    let num_rows = nested[0].len();

    let page = utils::build_plain_page(
        buffer,
        num_values,
        num_rows,
        array.clone().null_count(),
        repetition_levels_byte_length,
        definition_levels_byte_length,
        statistics,
        type_,
        options,
        Encoding::RleDictionary,
    )?;
    Ok(vec![Page::Data(page)])
}

fn serialize_keys<K: DictionaryKey>(
    array: &DictionaryArray<K>,
    type_: PrimitiveType,
    nested: &[Nested],
    statistics: Option<ParquetStatistics>,
    options: WriteOptions,
) -> PolarsResult<Vec<Page>> {
    if nested.len() == 1 {
        serialize_keys_flat(array, type_, statistics, options)
    } else {
        serialize_keys_nested(array, type_, nested, statistics, options)
    }
}

macro_rules! dyn_prim {
    ($from:ty, $to:ty, $array:expr, $options:expr, $type_:expr) => {{
        let values = $array.values().as_any().downcast_ref().unwrap();

        let buffer =
            primitive_encode_plain::<$from, $to>(values, EncodeNullability::new(false), vec![]);

        let stats: Option<ParquetStatistics> = if !$options.statistics.is_empty() {
            let mut stats = primitive_build_statistics::<$from, $to>(
                values,
                $type_.clone(),
                &$options.statistics,
            );
            stats.null_count = Some($array.null_count() as i64);
            Some(stats.serialize())
        } else {
            None
        };
        (
            DictPage::new(CowBuffer::Owned(buffer), values.len(), false),
            stats,
        )
    }};
}

pub fn array_to_pages<K: DictionaryKey>(
    array: &DictionaryArray<K>,
    type_: PrimitiveType,
    nested: &[Nested],
    options: WriteOptions,
    encoding: Encoding,
) -> PolarsResult<DynIter<'static, PolarsResult<Page>>> {
    match encoding {
        Encoding::PlainDictionary | Encoding::RleDictionary => {
            // write DictPage
            let (dict_page, mut statistics): (_, Option<ParquetStatistics>) = match array
                .values()
                .dtype()
                .to_logical_type()
            {
                ArrowDataType::Int8 => dyn_prim!(i8, i32, array, options, type_),
                ArrowDataType::Int16 => dyn_prim!(i16, i32, array, options, type_),
                ArrowDataType::Int32 | ArrowDataType::Date32 | ArrowDataType::Time32(_) => {
                    dyn_prim!(i32, i32, array, options, type_)
                },
                ArrowDataType::Int64
                | ArrowDataType::Date64
                | ArrowDataType::Time64(_)
                | ArrowDataType::Timestamp(_, _)
                | ArrowDataType::Duration(_) => dyn_prim!(i64, i64, array, options, type_),
                ArrowDataType::UInt8 => dyn_prim!(u8, i32, array, options, type_),
                ArrowDataType::UInt16 => dyn_prim!(u16, i32, array, options, type_),
                ArrowDataType::UInt32 => dyn_prim!(u32, i32, array, options, type_),
                ArrowDataType::UInt64 => dyn_prim!(u64, i64, array, options, type_),
                ArrowDataType::Float32 => dyn_prim!(f32, f32, array, options, type_),
                ArrowDataType::Float64 => dyn_prim!(f64, f64, array, options, type_),
                ArrowDataType::LargeUtf8 => {
                    let array = polars_compute::cast::cast(
                        array.values().as_ref(),
                        &ArrowDataType::LargeBinary,
                        Default::default(),
                    )
                    .unwrap();
                    let array = array.as_any().downcast_ref().unwrap();

                    let mut buffer = vec![];
                    binary_encode_plain::<i64>(array, EncodeNullability::Required, &mut buffer);
                    let stats = if options.has_statistics() {
                        Some(binary_build_statistics(
                            array,
                            type_.clone(),
                            &options.statistics,
                        ))
                    } else {
                        None
                    };
                    (
                        DictPage::new(CowBuffer::Owned(buffer), array.len(), false),
                        stats,
                    )
                },
                ArrowDataType::BinaryView => {
                    let array = array
                        .values()
                        .as_any()
                        .downcast_ref::<BinaryViewArray>()
                        .unwrap();
                    let mut buffer = vec![];
                    binview::encode_plain(array, EncodeNullability::Required, &mut buffer);

                    let stats = if options.has_statistics() {
                        Some(binview::build_statistics(
                            array,
                            type_.clone(),
                            &options.statistics,
                        ))
                    } else {
                        None
                    };
                    (
                        DictPage::new(CowBuffer::Owned(buffer), array.len(), false),
                        stats,
                    )
                },
                ArrowDataType::Utf8View => {
                    let array = array
                        .values()
                        .as_any()
                        .downcast_ref::<Utf8ViewArray>()
                        .unwrap()
                        .to_binview();
                    let mut buffer = vec![];
                    binview::encode_plain(&array, EncodeNullability::Required, &mut buffer);

                    let stats = if options.has_statistics() {
                        Some(binview::build_statistics(
                            &array,
                            type_.clone(),
                            &options.statistics,
                        ))
                    } else {
                        None
                    };
                    (
                        DictPage::new(CowBuffer::Owned(buffer), array.len(), false),
                        stats,
                    )
                },
                ArrowDataType::LargeBinary => {
                    let values = array.values().as_any().downcast_ref().unwrap();

                    let mut buffer = vec![];
                    binary_encode_plain::<i64>(values, EncodeNullability::Required, &mut buffer);
                    let stats = if options.has_statistics() {
                        Some(binary_build_statistics(
                            values,
                            type_.clone(),
                            &options.statistics,
                        ))
                    } else {
                        None
                    };
                    (
                        DictPage::new(CowBuffer::Owned(buffer), values.len(), false),
                        stats,
                    )
                },
                ArrowDataType::FixedSizeBinary(_) => {
                    let mut buffer = vec![];
                    let array = array.values().as_any().downcast_ref().unwrap();
                    fixed_binary_encode_plain(array, EncodeNullability::Required, &mut buffer);
                    let stats = if options.has_statistics() {
                        let stats = fixed_binary_build_statistics(
                            array,
                            type_.clone(),
                            &options.statistics,
                        );
                        Some(stats.serialize())
                    } else {
                        None
                    };
                    (
                        DictPage::new(CowBuffer::Owned(buffer), array.len(), false),
                        stats,
                    )
                },
                other => {
                    polars_bail!(
                        nyi =
                            "Writing dictionary arrays to parquet only support data type {other:?}"
                    )
                },
            };

            if let Some(stats) = &mut statistics {
                stats.null_count = Some(array.null_count() as i64)
            }

            // write DataPage pointing to DictPage
            let data_pages = serialize_keys(array, type_, nested, statistics, options)?;

            Ok(DynIter::new(
                std::iter::once(Page::Dict(dict_page))
                    .chain(data_pages)
                    .map(Ok),
            ))
        },
        _ => polars_bail!(nyi = "Dictionary arrays only support dictionary encoding"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::IntegerType;

    fn make_dict_array(num_values: usize, num_dict_values: usize) -> DictionaryArray<u32> {
        let keys = PrimitiveArray::<u32>::from_vec(
            (0..num_values)
                .map(|i| (i % num_dict_values) as u32)
                .collect(),
        );
        let values: Vec<Option<&str>> = (0..num_dict_values)
            .map(|i| Some(["A", "B", "C", "D", "E", "F", "G", "H"][i % 8]))
            .collect();
        let values = Utf8ViewArray::from_slice(&values);
        DictionaryArray::<u32>::try_new(
            ArrowDataType::Dictionary(
                IntegerType::UInt32,
                Box::new(ArrowDataType::Utf8View),
                false,
            ),
            keys,
            values.boxed(),
        )
        .unwrap()
    }

    fn make_options(data_page_size: Option<usize>) -> WriteOptions {
        WriteOptions {
            statistics: super::super::StatisticsOptions::empty(),
            compression: crate::parquet::compression::CompressionOptions::Uncompressed,
            version: super::super::Version::V1,
            data_page_size,
        }
    }

    fn make_type() -> PrimitiveType {
        PrimitiveType::from_physical(
            "col".into(),
            crate::parquet::schema::types::PhysicalType::Int32,
        )
    }

    /// Regression test for https://github.com/pola-rs/polars/issues/20141
    /// Verifies that dictionary-encoded columns produce multiple data pages
    /// when data_page_size is small.
    #[test]
    fn test_dictionary_multiple_data_pages() {
        let dict_array = make_dict_array(10000, 1);
        let options = make_options(Some(1));
        let type_ = make_type();

        let data_pages = serialize_keys_flat(&dict_array, type_, None, options).unwrap();

        assert!(
            data_pages.len() > 1,
            "Expected multiple data pages with small data_page_size, got {}",
            data_pages.len()
        );
    }

    #[test]
    fn test_dictionary_single_page_when_small() {
        // Small array should fit in one page with default page size
        let dict_array = make_dict_array(100, 1);
        let options = make_options(None); // Default page size (1MB)
        let type_ = make_type();

        let data_pages = serialize_keys_flat(&dict_array, type_, None, options).unwrap();

        assert_eq!(
            data_pages.len(),
            1,
            "Small array should fit in single page"
        );
    }

    #[test]
    fn test_dictionary_larger_dictionary() {
        // Test with multiple dictionary values (affects bit width estimation)
        let dict_array = make_dict_array(10000, 8);
        let options = make_options(Some(1));
        let type_ = make_type();

        let data_pages = serialize_keys_flat(&dict_array, type_, None, options).unwrap();

        assert!(
            data_pages.len() > 1,
            "Expected multiple data pages, got {}",
            data_pages.len()
        );
    }

    #[test]
    fn test_dictionary_with_nulls() {
        // Create array with nulls
        let keys = PrimitiveArray::<u32>::from(vec![Some(0), None, Some(0), None, Some(0)]);
        let values = Utf8ViewArray::from_slice([Some("A")]);
        let dict_array = DictionaryArray::<u32>::try_new(
            ArrowDataType::Dictionary(
                IntegerType::UInt32,
                Box::new(ArrowDataType::Utf8View),
                false,
            ),
            keys,
            values.boxed(),
        )
        .unwrap();

        let options = make_options(Some(1));
        let type_ = make_type();

        let data_pages = serialize_keys_flat(&dict_array, type_, None, options).unwrap();

        // Should complete without error
        assert!(!data_pages.is_empty());
    }

    #[test]
    fn test_dictionary_empty_array() {
        // Empty arrays should not panic (regression test for division by zero)
        let keys = PrimitiveArray::<u32>::from(Vec::<Option<u32>>::new());
        let values = Utf8ViewArray::from_slice(Vec::<Option<&str>>::new());
        let dict_array = DictionaryArray::<u32>::try_new(
            ArrowDataType::Dictionary(
                IntegerType::UInt32,
                Box::new(ArrowDataType::Utf8View),
                false,
            ),
            keys,
            values.boxed(),
        )
        .unwrap();

        let options = make_options(Some(1));
        let type_ = make_type();

        let data_pages = serialize_keys_flat(&dict_array, type_, None, options).unwrap();

        // Empty array should produce no data pages
        assert!(data_pages.is_empty());
    }
}
