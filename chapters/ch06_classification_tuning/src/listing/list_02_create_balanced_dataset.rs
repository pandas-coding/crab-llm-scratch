
use polars::prelude::{DataFrame, ChunkCompareEq, concat, IntoLazy, UnionArgs, Series};

/// Creating a balanced dataset
pub fn create_balanced_dataset(df: DataFrame) -> anyhow::Result<DataFrame> {
    // balance by undersampling
    let mask = df.column("label")?.i64()?.equal(1);
    let spam_subset = df.filter(&mask)?;
    let num_spam = spam_subset.shape().0;

    let mask = df.column("label")?.i64()?.equal(0);
    let ham_subset = df.filter(&mask)?;
    let n = Series::from_iter([num_spam as i32].iter());
    let undersampled_ham_subset = ham_subset.sample_n(&n, false , true, Some(1234u64))?;

    let balanced_df = concat(
        [
            spam_subset.clone().lazy(),
            undersampled_ham_subset.clone().lazy(),
        ],
        UnionArgs::default(),
    )?.collect()?;

    Ok(balanced_df)
}

#[cfg(test)]
mod tests {
    use crate::test_utils::sms_spam_df;
    use super::*;

    #[test]
    fn test_create_balanced_dataset() -> anyhow::Result<()> {
        let (df, num_spam) = sms_spam_df();
        let balanced_df = create_balanced_dataset(df)?;

        assert_eq!(balanced_df.shape(), (num_spam * 2usize, 2));
        Ok(())
    }
}
