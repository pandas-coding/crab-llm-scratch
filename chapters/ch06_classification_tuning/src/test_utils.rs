use std::path::PathBuf;
use polars::df;
use polars::frame::DataFrame;
use polars::prelude::ParquetWriter;
use tempfile::NamedTempFile;

pub fn sms_spam_df() -> (DataFrame, usize) {
    let df = df!(
        "sms"=> &[
                "Got it. Seventeen pounds for seven hundred ml – hope ok.",
                "Great News! Call FREEFONE 08006344447 to claim your guaranteed £1000 CASH or £2000 gift. Speak to a live operator NOW!",
                "No chikku nt yet.. Ya i'm free",
                "S:-)if we have one good partnership going we will take lead:)",
                "18 days to Euro2004 kickoff! U will be kept informed of all the latest news and results daily. Unsubscribe send GET EURO STOP to 83222."],
            "label"=> &[0_i64, 1, 0, 0, 1],
    ).unwrap();
    (df, 2usize)
}

pub fn test_parquet_path() -> PathBuf {
    let (mut df, _num_spam) = sms_spam_df();
    let test_file = NamedTempFile::new().unwrap();
    ParquetWriter::new(&test_file).finish(&mut df).unwrap();
    test_file.into_temp_path().keep().unwrap()
}
