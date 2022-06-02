from processor.tokenize_cluster import ClusterTokenizer

ClusterTokenizer(
    data_dir='data/ListContUni/zhihu',
    cluster_json='data/cluster/zhihu-n1000.json',
).tokenize()
