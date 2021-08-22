import get_vectors
import pickle
import json

path = "/data/zhangxy/hello_pytorch/classifier/data/"

if __name__ == "__main__":
    # get_vectors.get_best_model()  # 如果没有model, 则训练并保存model
    rev_mappings = json.load(open(path + 'mapping.json', 'r'))
    model = pickle.load(open(path + 'gaussianNB.pickle', 'rb'))
    print("mappings和模型加载完毕！开始预测")

    text = "监控模块组串没有电流经过"
    text_vec = get_vectors.get_single_vec(text)
    pre_label = model.predict([text_vec]).tolist()
    print("测试数据的维修类别是:{}，维修方式是:{}".format(pre_label[0], rev_mappings[str(pre_label[0])]))
