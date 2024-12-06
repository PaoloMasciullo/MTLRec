import argparse
import warnings
import os
import zipfile
import requests
import io

warnings.filterwarnings("ignore")


def download_ebnerd_dataset(dataset_size, embedding_type, data_path, train_path, test_path=None):
    dataset_url = f"https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_{dataset_size}.zip"
    test_url = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip"
    contrast_emb_url = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_contrastive_vector.zip"
    bert_emb_url = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/google_bert_base_multilingual_cased.zip"
    roberta_emb_url = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/FacebookAI_xlm_roberta_base.zip"
    image_emb_url = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_image_embeddings.zip"

    print(f"Getting Ebnerd {dataset_size} from : {dataset_url} ..")
    response_dataset = requests.get(dataset_url)
    with zipfile.ZipFile(io.BytesIO(response_dataset.content)) as zip_ref:
        zip_ref.extract("articles.parquet", path=train_path)
        zip_ref.extract("train/history.parquet", path=data_path)
        zip_ref.extract("train/behaviors.parquet", path=data_path)

        zip_ref.extract("validation/history.parquet", path=data_path)
        zip_ref.extract("validation/behaviors.parquet", path=data_path)

    if test_path:
        print(f"Getting Ebnerd test from : {test_url} ..")

        response_test = requests.get(test_url)

        with zipfile.ZipFile(io.BytesIO(response_test.content)) as zip_ref:
            zip_ref.extract("ebnerd_testset/articles.parquet")
            zip_ref.extract("ebnerd_testset/test/history.parquet")
            zip_ref.extract("ebnerd_testset/test/behaviors.parquet")
        os.rename("./ebnerd_testset/test", test_path)
        os.rename("./ebnerd_testset/articles.parquet", os.path.join(test_path, "articles.parquet"))
        os.removedirs("./ebnerd_testset")

    print(f"Getting news image embeddings from : {image_emb_url} ..")
    response_image_emb = requests.get(image_emb_url)
    with zipfile.ZipFile(io.BytesIO(response_image_emb.content)) as zip_ref:
        zip_ref.extract("Ekstra_Bladet_image_embeddings/image_embeddings.parquet")
    os.rename("./Ekstra_Bladet_image_embeddings/image_embeddings.parquet", data_path + "/image_embeddings.parquet")
    os.removedirs("./Ekstra_Bladet_image_embeddings")
    if embedding_type == "contrastive":
        print(f"Getting news contrastive embeddings from : {contrast_emb_url} ..")
        response_contrast_emb = requests.get(contrast_emb_url)
        with zipfile.ZipFile(io.BytesIO(response_contrast_emb.content)) as zip_ref:
            zip_ref.extract("Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet")
        os.rename("Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet",
                  data_path + "/contrastive_vector.parquet")
        os.removedirs("./Ekstra_Bladet_contrastive_vector")
    elif embedding_type == "bert":
        print(f"Getting news bert embeddings from : {bert_emb_url} ..")
        response_bert_emb = requests.get(bert_emb_url)
        with zipfile.ZipFile(io.BytesIO(response_bert_emb.content)) as zip_ref:
            zip_ref.extract("google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet")
        os.rename("google_bert_base_multilingual_cased/bert_base_multilingual_cased.parquet",
                  data_path + "/bert_vector.parquet")
        os.removedirs("./google_bert_base_multilingual_cased")
    elif embedding_type == "roberta":
        print(f"Getting news roberta embeddings from : {roberta_emb_url} ..")
        response_roberta_emb = requests.get(roberta_emb_url)
        with zipfile.ZipFile(io.BytesIO(response_roberta_emb.content)) as zip_ref:
            zip_ref.extract("FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet")
        os.rename("FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet", data_path + "/roberta_vector.parquet")
        os.removedirs("./FacebookAI_xlm_roberta_base")
    else:
        raise Exception("Embedding type not available")

    print(f"Dataset Ebnerd {dataset_size} downloaded!")

if __name__ == '__main__':
    ''' 
    Usage: 
    python download_data.py --size {dataset_size} --data_folder {data_path} [--test] 
                                --embedding_size [64|128|256] --embedding_type [contrastive|bert|roberta]
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str, default='small', help='The size of the dataset to download')
    parser.add_argument('--data_folder', type=str, default='./', help='The folder in which data will be stored')
    parser.add_argument('--tag', type=str, default='x1', help='The tag of the preprocessed dataset to save')
    parser.add_argument('--test', action="store_true", help='Use this flag to download the test set (default no)')
    parser.add_argument('--embedding_type', type=str, default='roberta',
                        help='The embedding type you want to use')

    args = vars(parser.parse_args())
    dataset_size = args['size']
    data_folder = args['data_folder']
    embedding_type = args['embedding_type']
    tag = args['tag']
    # insert a check, if data aren't in the repository, download them
    dataset_path = os.path.join(data_folder, 'Ebnerd_' + dataset_size)
    # Check if 'Ebnerd_{dataset_size}' folder exists
    if os.path.isdir(dataset_path):
        print(f"Folder '{dataset_path}' exists.")
        # Check if 'Ebnerd_{dataset_size}' folder is empty
        if not os.listdir(dataset_path):
            print(f"Folder '{dataset_path}' is empty. Downloading the dataset...")
            # download the dataset
            if args['test']:
                print("Downloading the test set")
                download_ebnerd_dataset(dataset_size, embedding_type, dataset_path, dataset_path + '/train/', dataset_path + '/test/')
            else:
                print("Not Downloading the test set")
                download_ebnerd_dataset(dataset_size, embedding_type, dataset_path, dataset_path + '/train/')
        else:
            print(f"Folder '{dataset_path}' is not empty. The dataset is already downloaded")
            # end, we will not download anything
    else:
        print(f"Folder '{dataset_path}' does nost exist. Creating it now.")

        # Create the 'ebnerd_demo' folder
        os.makedirs(dataset_path)
        print(f"Folder '{dataset_path}' has been created.")
        # now we will download the dataset here
        print("Downloading the data set")
        download_ebnerd_dataset(dataset_size, embedding_type, dataset_path, dataset_path + '/train/', dataset_path + '/test/')
