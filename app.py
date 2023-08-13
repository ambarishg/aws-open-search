import streamlit as st
from config import *

import sagemaker
import json

from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import boto3
from requests_aws4auth import AWS4Auth
from sentence_transformers import SentenceTransformer
import pandas as pd

def query_endpoint_with_json_payload(encoded_json, endpoint_name, content_type="application/json"):
    client = boto3.client("runtime.sagemaker")
    response = client.invoke_endpoint(
        EndpointName=endpoint_name, ContentType=content_type, Body=encoded_json
    )
    return response


st.set_page_config(page_title="Search Engine", page_icon="🔍")

st.header('Search Engine - Document')


user_input = st.text_area('Enter your question here:')

if st.button('Make recommendation'):
    region = region 
    index_name = index_name
    service = service
    aos_host = aos_host

    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, region)
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)


    aos_client = OpenSearch(
    hosts = [{'host': aos_host, 'port': 443}],
    http_auth = auth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection
)
    
    model = SentenceTransformer(MODEL_NAME)
    xq = model.encode([user_input],convert_to_tensor=True)
    xq_list = xq.tolist()

    query={
    "size": 10,
    "query": {
        "knn": {
            "text_vector":{
                "vector":xq_list[0],
                "k":10
            }
        }
    }
}

    res = aos_client.search(index=index_name, 
                       body=query,
                       stored_fields=["text"])
    
    
    contexts =""
    counter = 0

    for hit in res['hits']['hits']:
        if counter > 3:
            break
        contexts +=  hit['fields']['text'][0]+"\n---\n"
        counter += 1

    prompt = """Answer based on context:\n\n{context}\n\n{question}"""

    text_input = prompt.replace("{context}", contexts)
    text_input = text_input.replace("{question}", user_input)

    # hyperparameters for llm
    payload = {
        "inputs": text_input,
        "parameters": {
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.1,
            "max_new_tokens": 1024,
            "stop": ["<|endoftext|>", "</s>"],
        },
    }

    response2 = query_endpoint_with_json_payload(
        json.dumps(payload).encode("utf-8"), llm_endpoint)
    model_predictions = json.loads(response2["Body"].read())
    st.write(model_predictions["predictions"][0]["generated_text"])



    



