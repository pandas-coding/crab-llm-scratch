
use polars::prelude::{DataFrame, Series};

/// [Listing 6.3] Splitting the dataset
pub fn random_split(
    df: &DataFrame,
    train_frac: f32,
    validation_frac: f32,
) -> anyhow::Result<(DataFrame, DataFrame, DataFrame)> {
    let frac = Series::from_iter([1f32].iter());
    let shuffled_df = df.sample_frac(&frac, false, true, Some(123_u64))?;

    let df_size = df.shape().0;
    let train_size = (df.shape().0 as f32 * train_frac) as usize;
    let validation_size = (df.shape().0 as f32 * validation_frac) as usize;

    let train_df = shuffled_df.slice(0_i64, train_size);
    let validation_df = shuffled_df.slice(train_size as i64, validation_size);
    let test_df = shuffled_df.slice(
        (train_size + validation_size) as i64,
        df_size - train_size - validation_size,
    );
    Ok((train_df, validation_df, test_df))
}

#[cfg(test)]
mod tests {
    use crate::test_utils::sms_spam_df;
    use super::*;

    #[test]
    fn test_random_split() -> anyhow::Result<()> {
        let (df, _num_spam) = sms_spam_df();
        let train_frac = 0.4f32;
        let validation_frac = 0.4f32;
        let test_frac = 1f32 - train_frac - validation_frac;
        let (train_df, validation_df, test_df) = random_split(&df, train_frac, validation_frac)?;

        assert_eq!(
            train_df.shape(),
            ((train_frac * df.shape().0 as f32) as usize, 2)
        );
        assert_eq!(
            validation_df.shape(),
            ((validation_frac * df.shape().0 as f32) as usize, 2)
        );
        assert_eq!(
            test_df.shape(),
            ((test_frac * df.shape().0 as f32) as usize, 2)
        );

        Ok(())
    }
}