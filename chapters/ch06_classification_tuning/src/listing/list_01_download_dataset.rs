use std::fs::{create_dir_all, remove_file, rename, File};
use std::io;
use std::path::{Path, PathBuf};
use zip::ZipArchive;

pub const URL: &str = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip";
pub const ZIP_PATH: &str = "data/sms_spam_collection.zip";
pub const EXTRACTED_PATH: &str = "sms_spam_collection";
pub const EXTRACTED_FILENAME: &str = "SMSSpamCollection";
pub const PARQUET_URL: &str = "https://huggingface.co/datasets/ucirvine/sms_spam/resolve/main/plain_text/train-00000-of-00001.parquet?download=true";
pub const PARQUET_FILENAME: &str = "train-00000-of-00001.parquet";

/// Download spam parquet dataset from HuggingFace
pub fn download_smsspam_parquet(url: &str) -> anyhow::Result<()> {
    let resp = reqwest::blocking::get(url)?;
    let content = resp.bytes()?;

    let mut out_path = PathBuf::from("data");
    out_path.push(PARQUET_FILENAME);
    let mut out_file = File::create(out_path)?;
    io::copy(&mut content.as_ref(), &mut out_file)?;
    Ok(())
}

pub fn download_and_unzip_spam_data(
    url: &str,
    zip_path: &str,
    extracted_path: &str,
) -> anyhow::Result<()> {
    let resp = reqwest::blocking::get(url)?;
    let content = resp.bytes()?;
    let mut out = File::create(zip_path)?;
    io::copy(&mut content.as_ref(), &mut out)?;

    unzip_file(zip_path)?;

    // rename file to add .tsv extension
    let original_file_path = Path::new("data").join(EXTRACTED_FILENAME);
    let data_file_path = original_file_path.with_extension("tsv");
    rename(original_file_path, data_file_path)?;

    // remove zip file and readme file
    let readme_file: PathBuf = ["data", "readme"].iter().collect();
    remove_file_if_exists(readme_file)?;
    remove_file_if_exists(ZIP_PATH)?;
    Ok(())
}

/// Helper function to unzip file using `zip::ZipArchive`
fn unzip_file(filename: &str) -> anyhow::Result<()> {
    let mut archive = ZipArchive::new(File::open(filename)?)?;

    (0..archive.len()).try_for_each(|i| {
        let mut entry = archive.by_index(i)?;

        let outpath = match entry.enclosed_name() {
            Some(path) => PathBuf::from("data").join(path),
            None => return Ok(()), // 没有安全路径就跳过
        };

        let comment = entry.comment();
        (!comment.is_empty()).then(|| println!("File {i} comment: {comment}"));

        if entry.is_dir() {
            create_dir_all(&outpath)?;
            println!("File {} extracted to \"{}\"", i, outpath.display());
            return Ok(());
        }

        let size = entry.size();

        // 父目录如果不存在则创建，0/1 个元素用 into_iter + filter + try_for_each 组合处理
        outpath
            .parent()
            .into_iter()
            .filter(|p| !p.exists())
            .try_for_each(create_dir_all)?;

        {
            // IO 必须使用可变引用，这里的 mut 是难以避免的
            let mut outfile = File::create(&outpath)?;
            io::copy(&mut entry, &mut outfile)?;
        }

        println!(
            "File {} extracted to \"{}\" ({} bytes)",
            i,
            outpath.display(),
            size
        );

        Ok(())
    })
}

/// A wrapper for std::fs::remove_file that passes on any ErrorKind::NotFound
fn remove_file_if_exists<P: AsRef<Path>>(fileName: P) -> anyhow::Result<()> {
    match remove_file(fileName) {
        Ok(_) => Ok(()),
        Err(err) => match err.kind() {
            io::ErrorKind::NotFound => Ok(()),
            _ => Err(anyhow::Error::from(err)),
        }
    }
}
