import pandas as pd
from PIL import Image
import numpy as np
import io
import os

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Pyspark
import findspark
findspark.init()
import pyspark


from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, pandas_udf
from pyspark import SparkContext, SparkConf


import pyarrow

from contextlib import contextmanager
import time 

import boto3

from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
import seaborn as sns
import matplotlib.pyplot as plt

print('pyspark ==', pyspark.__version__)
import py4j
print('py4j ==', py4j.__version__)
print('tf ==', tf.__version__)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# Check content of S3 bucket 
connexion = boto3.client('s3')
contents = connexion.list_objects(Bucket='p8-devenv')['Contents']
for file in contents:
    print(file['Key']) 



# Configure Spark Session to work with AWS S3 bucket
with timer('Total Apps time'):
    os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages com.amazonaws:aws-java-sdk-pom:1.10.34,org.apache.hadoop:hadoop-aws:2.7.2 pyspark-shell'

    accessKeyId='<YOUR_ACC_KEY>'
    secretAccessKey='<YOUR_SEC_KEY>'

    spark = (SparkSession
            .builder.master('local[*]')
            .appName('P8 - Déployez un modèle dans le cloud')
            .config('spark.hadoop.fs.s3a.access.key', accessKeyId)
            .config('spark.hadoop.fs.s3a.secret.key', secretAccessKey) 
            .config('spark.hadoop.fs.s3a.impl', 'org.apache.hadoop.fs.s3a.S3AFileSystem') 
            .getOrCreate()
            )

    sc = spark.sparkContext
    sc.setSystemProperty('com.amazonaws.services.s3.enableV4', 'true')
    sc._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3.eu-west-1.amazonaws.com")
    with timer('Load Data'):
        # Path through image (bucket S3)
        data_source = 's3a://p8-devenv/data/*'
        output_uri = 's3a://p8-devenv/resultats/'

        # Binaryfile format
        data = spark.read.format("binaryFile") \
          .load(data_source)

        # Extraction of the label from the path string
        image_df = data.withColumn('label', split(col('path'), '/').getItem(6))
        image_df = image_df.select('path', 'content', 'label')

    with timer('Load Model'):
        # Export CNN model 
        model = ResNet50(include_top=False)

        # Broadcast model weights to the workers
        bc_model_weights = spark.sparkContext.broadcast(model.get_weights())

    def model_fn():
        """
        Returns a ResNet50 model with top layer removed and broadcasted pretrained weights.
        """
        model = ResNet50(weights=None, include_top=False)
        model.set_weights(bc_model_weights.value)
        return model

    def preprocess(content):
        """
        Preprocesses raw image bytes for prediction.
        """
        img = Image.open(io.BytesIO(content)).resize([224, 224])
        arr = img_to_array(img)
        return preprocess_input(arr)

    def featurize_series(model, content_series):
        """
        Featurize a pd.Series of raw images using the input model.
        :return: a pd.Series of image features
        """
        input = np.stack(content_series.map(preprocess))
        preds = model.predict(input)
        # For some layers, output features will be multi-dimensional tensors.
        # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
        output = [p.flatten() for p in preds]
        return pd.Series(output)


    from typing import Iterator
    @pandas_udf('array<float>')
    def featurize_udf(content_series_iter:Iterator[pd.Series]) -> Iterator[pd.Series]:
        '''
        This method is a Scalar Iterator pandas UDF wrapping our featurization function.
        The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).
        
        :param content_series_iter: This argument is an iterator over batches of data, where each batch
                                    is a pandas Series of image data.
        '''
        # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
        # for multiple data batches.  This amortizes the overhead of loading big models.
        model = model_fn()
        for content_series in content_series_iter:
            yield featurize_series(model, content_series)


    # Pandas UDFs on large records (e.g., very large images) can run into Out Of Memory (OOM) errors.
    # To fix this error we can try reducing the Arrow batch size via `maxRecordsPerBatch`.
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")

    with timer('Featurization'):
        # We can now run featurization on our entire Spark DataFrame.
        # NOTE: This can take a long time (if we are taking a lot of fruits image) since it applies a large model to the full dataset.
        features_df = image_df.select(col("path"), col("label"), featurize_udf("content").alias("features"))


    ## Diension Reduction

    def pca_transformation(df, col_name:str, n_components:int=10, variance_plot:bool=False):
        
        """
        Apply PCA to all the image to reduce the number of features for the model

        Paramètres:
        df(pyspark dataFrame): dataframe with all the image information
        col_name: the dataframe column which to aapply the PCA
        n_components(int): nombre de dimensions à conserver
        """

        # Image data are convert into dense vector format
        to_vector_udf = udf(lambda r: Vectors.dense(r), VectorUDT())
        df = df.withColumn('X_vectors', to_vector_udf(col_name))

        # Fitting the PCA class
        pca = PCA(k=n_components, inputCol='X_vectors', outputCol='X_vectors_pca')
        model_pca = pca.fit(df)

        # Feature PCA transformation
        df = model_pca.transform(df)

        if variance_plot == True:
            # Show the Explained Variance of the model 
            var = model_pca.explainedVariance.cumsum()
            plt.figure(figsize=(15, 10))
            sns.set_context(context='poster', font_scale=0.8)
            sns.lineplot(x=[i for i in range(n_components + 1)], y=np.insert(var,0,0)*100, color='deepskyblue')
            plt.xlabel('PCs')
            plt.ylabel('Variance (%)')
            plt.ylim(0,100)
            plt.xlim(left=0)
            plt.show()      

        return df

    with timer('Dimension Reduction - PCA'):
        df_final = pca_transformation(df=features_df, col_name='features', n_components=10)
        df_final.head()

    # Export results to bucket S3
    with timer('Write final results'):
        df_final = df_final.select('path', 'label', 'X_vectors_pca')

        df_final.write.mode('overwrite').parquet(output_uri + 'results')
    
        df_final_pandas = df_final.toPandas()
        df_final_pandas.to_csv(output_uri + 'results.csv', index=False)
