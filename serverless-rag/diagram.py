"""Architecture diagram for the serverless-rag benchmark project."""

from diagrams import Diagram, Edge
from diagrams.custom import Custom

ICONS = "/Users/deburky/Documents/AWS-Icons/Icon-package/Architecture-Service-Icons_01302026"

LAMBDA = f"{ICONS}/Arch_Compute/64/Arch_AWS-Lambda_64.png"
AURORA = f"{ICONS}/Arch_Databases/64/Arch_Amazon-Aurora_64.png"
DYNAMO = f"{ICONS}/Arch_Databases/64/Arch_Amazon-DynamoDB_64.png"
APIGW = f"{ICONS}/Arch_Networking-Content-Delivery/64/Arch_Amazon-API-Gateway_64.png"
S3 = f"{ICONS}/Arch_Storage/64/Arch_Amazon-Simple-Storage-Service_64.png"
SSM = f"{ICONS}/Arch_Management-Tools/64/Arch_AWS-Systems-Manager_64.png"

FONT = "Amazon Ember"

graph_attr = {
    "fontname": FONT,
    "fontsize": "20",
    "fontcolor": "black",
    "bgcolor": "white",
    "splines": "false",
    "nodesep": "0.05",
    "ranksep": "0.8",
    "labelloc": "t",
    "pad": "0.5",
    "dpi": "200",
}
node_attr = {
    "fontname": FONT,
    "fontsize": "11",
    "fontcolor": "black",
    "imagescale": "true",
    "width": "1.0",
    "height": "0.7",
    "margin": "0,0",
}
edge_attr = {
    "fontname": FONT,
    "fontsize": "9",
    "fontcolor": "black",
}

with Diagram(
    "Serverless RAG: DynamoDB vs Aurora pgvector",
    filename="serverless-rag/docs/architecture",
    outformat="png",
    show=False,
    direction="LR",
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr,
):
    client = Custom("API Gateway\n(HTTP API)", APIGW)
    s3 = Custom("S3 Model Store\n(MiniLM / Qwen)", S3)
    ssm = Custom("SSM\naurora-endpoint", SSM)

    ingest = Custom("Ingest\nPOST /ingest", LAMBDA)
    query = Custom("Query\nGET /query", LAMBDA)
    answer = Custom("Answer\nGET /answer", LAMBDA)

    dynamo = Custom("DynamoDB\nbrute-force O(n)", DYNAMO)
    aurora = Custom("Aurora V2\npgvector IVFFlat", AURORA)

    # API → Lambdas
    client >> ingest
    client >> query
    client >> answer

    # S3 cold-start
    s3 >> Edge(style="dashed", color="gray") >> ingest
    s3 >> Edge(style="dashed", color="gray") >> query
    s3 >> Edge(style="dashed", color="gray") >> answer

    # SSM config
    ssm >> Edge(style="dotted", color="gray") >> ingest

    # Backend writes / queries
    ingest >> Edge(label="dual write") >> dynamo
    ingest >> Edge(label="dual write") >> aurora
    query >> Edge(label="route") >> dynamo
    query >> Edge(label="route") >> aurora
    answer >> dynamo
    answer >> aurora
