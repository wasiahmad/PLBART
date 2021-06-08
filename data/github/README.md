### Little Guide to download Github data from Google Big Query

- Create a Google platform account ( you will have around $300 given for free , that is sufficient for Github)
- Create a Google Big Query project here
- In this project, create a dataset
- In this dataset, create one table per programming language. The results of each SQL request (one per language) will be stored in these tables.
- Before running your SQL request, make sure you change the query settings to save the query results in the dedicated table (more -> Query Settings -> Destination -> table for query results -> put table name)
- Run your SQL request (one per language and dont forget to change the table for each request)
- Export your results to google Cloud :
  - In google cloud storage, create a bucket and a folder per language into it
  - Export your table to this bucket ( EXPORT -> Export to GCS -> export format JSON , compression GZIP)
- To download the bucket on your machine, use the API gsutil:
  - pip install gsutil
  - gsutil config -> to config gsutil with your google account
  - gsutil -m cp -r gs://name_of_bucket/name_of_folder . -> copy your bucket on your machine

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

#### Reference

[https://github.com/facebookresearch/TransCoder#little-guide-to-download-github-from-google-big-query](https://github.com/facebookresearch/TransCoder#little-guide-to-download-github-from-google-big-query)

### Preprocessing Github data
    
```
python ../preprocessing/preprocess.py \
    --root . \
    --lang1 java \
    --lang2 python \
    --test_size 10000;
```

After preprocessing, in each directory (java/ and python/), the following files should be created.

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
