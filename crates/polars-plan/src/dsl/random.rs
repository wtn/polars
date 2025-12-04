use polars_core::random::get_global_random_u64;

use super::*;

impl Expr {
    pub fn shuffle(self, seed: Option<u64>) -> Self {
        let seed = seed.or_else(|| Some(get_global_random_u64()));
        self.map_unary(FunctionExpr::Random {
            method: RandomMethod::Shuffle,
            seed,
        })
    }

    pub fn sample_n(
        self,
        n: Expr,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> Self {
        let seed = seed.or_else(|| Some(get_global_random_u64()));
        self.map_binary(
            FunctionExpr::Random {
                method: RandomMethod::Sample {
                    is_fraction: false,
                    with_replacement,
                    shuffle,
                },
                seed,
            },
            n,
        )
    }

    pub fn sample_frac(
        self,
        frac: Expr,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> Self {
        let seed = seed.or_else(|| Some(get_global_random_u64()));
        self.map_binary(
            FunctionExpr::Random {
                method: RandomMethod::Sample {
                    is_fraction: true,
                    with_replacement,
                    shuffle,
                },
                seed,
            },
            frac,
        )
    }
}
