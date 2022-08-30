import utils_xml
from tqdm import tqdm

def fun4(feature_path, wsi_path, xml_path):
    img_maker = utils_xml.make_graph_img()
    img_maker.read_csv(feature_path, wsi_path, xml_path, omit_edge=['N-N', 'T-N', 'I-N', 'S-N'], omit_cell=[5])
    img_maker.make_img()
    img_maker.write_img()