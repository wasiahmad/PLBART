### Little Guide to download Github data from Google Big Query

**[Update]** We provide a step-by-step guide in
this [PDF](https://github.com/wasiahmad/PLBART/blob/main/data/bigquery_guide.pdf)
.

The following guide is borrowed
from [here](https://github.com/facebookresearch/TransCoder#little-guide-to-download-github-from-google-big-query). We
followed this guide to download the Github data in this project.

- Create a Google platform account (you will be given $300 free credit that is sufficient to download the Github data).
- Create a Google Big Query project [here](https://console.cloud.google.com/projectselector2/bigquery).
- In this project, create a dataset.
- In this dataset, create one table per programming language. The results of each SQL request (one per language) will be
  stored in these tables.
- Before running your SQL request, make sure you change the query settings to save the query results in the dedicated
  table (more -> Query Settings -> Destination -> table for query results -> put table name)
- Run your SQL request (one per language and dont forget to change the table for each request)
- Export your results to google Cloud :
    - In google cloud storage, create a bucket and a folder per language into it
    - Export your table to this bucket (EXPORT -> Export to GCS -> export format JSON , compression GZIP)
- To download the bucket on your machine, use the API gsutil:
    - `pip install gsutil`
    - `gsutil config -> to config gsutil with your google account`
    - copy your bucket on your machine -> `gsutil -m cp -r gs://name_of_bucket/name_of_folder .`

**NOTE** We set `name_of_folder` to `java` and `python` for respective source files.

#### Example of query for python :

``` 
SELECT 
    f.repo_name,
    f.ref,
    f.path,
    c.copies,
    c.content
FROM `bigquery-public-data.github_repos.files` as f
JOIN `bigquery-public-data.github_repos.contents` as c on f.id = c.id
WHERE 
    NOT c.binary
    AND f.path like '%.py'
```

#### Helpful links

- [https://console.cloud.google.com/marketplace/product/github/github-repos](https://console.cloud.google.com/marketplace/product/github/github-repos)
- [https://cloud.google.com/bigquery/public-data](https://cloud.google.com/bigquery/public-data)
- [https://cloud.google.com/bigquery/docs/quickstarts/quickstart-web-ui](https://cloud.google.com/bigquery/docs/quickstarts/quickstart-web-ui)
- [https://hoffa.medium.com/github-on-bigquery-analyze-all-the-code-b3576fd2b150](https://hoffa.medium.com/github-on-bigquery-analyze-all-the-code-b3576fd2b150)

### Preprocessing Github data

#### Dependencies

- [submitit](https://pypi.org/project/submitit/) (to run the preprocessing pipeline on remote machine)
- [libclang](https://pypi.org/project/clang/) (for C++ tokenization)
- [sacrebleu](https://pypi.org/project/sacrebleu/) (`pip install sacrebleu=="1.2.11"`)

#### Preprocess

```
python -m preprocessing.preprocess \
    path_2_github_data \
    --lang1 java \
    --lang2 python \
    --test_size 10000;
```

After preprocessing, in each directory (`java/` and `python/`), the following files should be created.

```
train.0.functions_class.tok
train.0.functions_standalone.tok
train.1.functions_class.tok
train.1.functions_standalone.tok
train.2.functions_standalone.tok
train.2.functions_class.tok
train.3.functions_class.tok
train.3.functions_standalone.tok
train.4.functions_class.tok
train.4.functions_standalone.tok
train.5.functions_class.tok
train.5.functions_standalone.tok
train.6.functions_class.tok
train.6.functions_standalone.tok
train.7.functions_class.tok   
train.7.functions_standalone.tok  
valid.functions_class.tok
valid.functions_standalone.tok
test.functions_class.tok
test.functions_standalone.tok
```



