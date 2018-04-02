from struct import unpack, calcsize 
import logging

logging.basicConfig(format='%(message)s', filename='log1',
                    filemode='w', level=logging.INFO)
logging.getLogger(__name__).addHandler(logging.NullHandler())


class LearnerModelParam(object):
    def __init__(self,  base_score):
        self.base_score = base_score

    @staticmethod
    def load(buffer):
        # >>> LearnerModelParam
        # float base_score
        fmt = 'f'
        offset = 0
        length = calcsize(fmt)
        (base_score,) = unpack(fmt, buffer[offset:offset+length])
        logging.debug("base_score:{:.6f}".format(base_score))
        offset += length

        # int padding[33]
        fmt = 'i'
        length = calcsize(fmt) * 33
        offset += length
        # <<< LearnerModelParam
        return (offset, LearnerModelParam(base_score))


class GBTreeModelParam(object):
    def __init__(self, num_trees, num_feature, num_output_group):
        self.num_trees = num_trees
        self.num_feature = num_feature
        self.num_output_group = num_output_group

    @staticmethod
    def load(buffer):
        # >>> GBTreeModelParam
        # int num_trees
        offset = 0
        fmt = 'iiiiQi'
        length = calcsize(fmt)
        (num_trees, _, num_feature, _, _, num_output_group) = unpack(
            fmt, buffer[offset:offset+length])
        logging.debug("num_trees:{}".format(num_trees))
        logging.debug("num_feature:{}".format(num_feature))
        logging.debug("num_output_group:{}".format(num_output_group))
        offset += length

        # int pad3[33]
        fmt = 'i'
        length = calcsize(fmt) * 33
        offset += length

        # <<< GBTreeModelParam

        return (offset, GBTreeModelParam(num_trees, num_feature, num_output_group))


class TreeParam(object):
    def __init__(self,  num_roots, num_nodes, num_deleted, size_leaf_vector):
        self.num_roots = num_roots
        self.num_nodes = num_nodes
        self.num_deleted = num_deleted
        self.size_leaf_vector = size_leaf_vector

    @staticmethod
    def load(buffer):
        # >>> TreeParam
        fmt = 'iiiiii'
        offset = 0
        length = calcsize(fmt)
        (num_roots, num_nodes, num_deleted, _, _, size_leaf_vector) = unpack(
            fmt, buffer[offset:offset+length])
        offset += length
        logging.debug("num_roots:{}".format(num_roots))
        logging.debug("num_nodes:{}".format(num_nodes))
        logging.debug("num_deleted:{}".format(num_deleted))
        logging.debug("size_leaf_vector:{}".format(size_leaf_vector))
        # int reserved[31]
        fmt = 'i'
        length = calcsize(fmt) * 31
        offset += length
        # <<< TreeParam
        return (offset, TreeParam(num_roots, num_nodes, num_deleted, size_leaf_vector))


class XGBTreeNode(object):
    def __init__(self, is_leaf, parent, cleft, cright, split_index, leaf_value_or_split_cond):
        self.is_leaf = is_leaf
        self.parent = parent
        self.cleft = cleft
        self.cright = cright
        self._split_index = split_index
        if is_leaf:
            self.leaf_value = leaf_value_or_split_cond
        else:
            self.split_cond = leaf_value_or_split_cond

    def split_index(self):
        return self._split_index & ((1 << 31) - 1)

    def cdefault(self):
        default_left = (self._split_index >> 31) != 0
        logging.info("default_left:{}".format(default_left))
        if default_left:
            return self.cleft
        else:
            return self.cright

    @staticmethod
    def load(buffer):
        # >>> Node
        fmt = 'iiiIf'
        offset = 0
        length = calcsize(fmt)
        (parent, cleft, cright, split_index, leaf_value_or_split_cond) = unpack(
            fmt, buffer[offset:offset+length])
        logging.debug("parent:{}".format(parent))
        logging.debug("cleft:{}".format(cleft))
        logging.debug("cright:{}".format(cright))
        logging.debug("split_index:{}".format(split_index))
        is_leaf = (cleft == -1 )
        if is_leaf:
            logging.debug("leaf_value:{:.6f}".format(
            leaf_value_or_split_cond))
        else:
            logging.debug("split_cond:{:.6f}".format(
            leaf_value_or_split_cond))
        offset += length
        # <<< Node
        return (offset, XGBTreeNode(is_leaf, parent, cleft, cright, split_index, leaf_value_or_split_cond))


class XGBTree(object):
    def __init__(self, tree_param, nodes):
        self.tree_param = tree_param
        self.nodes = nodes

    def get_leaf_index(self, feat):
        nid = 0  # root_id
        logging.info("root_id:{}".format(nid))
        while not self.nodes[nid].is_leaf:
            nid = self.get_next(nid, feat)
        return nid

    def get_next(self, nid, feat):
        split_index = self.nodes[nid].split_index()
        logging.info("split_index:{}".format(split_index))
        if feat.is_missing(split_index):
            logging.info("missing")
            nid = self.nodes[nid].cdefault()
            # logging.info("cdefault:{}".format(nid))
        else:
            fvalue = feat.fvalue(split_index) 
            split_cond = self.nodes[nid].split_cond 
            logging.info("fvalue:{:.6f}".format(fvalue))
            logging.info("split_cond:{:.6f}".format(split_cond))
            if  fvalue < split_cond:
                nid = self.nodes[nid].cleft
                logging.info("cleft:{}".format(nid))
            else:
                nid = self.nodes[nid].cright
                logging.info("cright:{}".format(nid))
        return nid

    @staticmethod
    def load(buffer):
        offset = 0
        # >>> TreeParam
        length, tree_param = TreeParam.load(buffer[offset:])
        offset += length
        # <<< TreeParam

        # >>> nodes
        nodes = []
        print("node: {}".format(tree_param.num_nodes))
        for i in range(tree_param.num_nodes):
            length, node = XGBTreeNode.load(buffer[offset:])
            offset += length
            nodes.append(node)
        # >>> nodes

        # CONSUME_BYTES(fi, (3 * sizeof(bst_float) + sizeof(int)) * param.num_nodes);
        offset += (4 * 4) * tree_param.num_nodes

        if tree_param.size_leaf_vector != 0:
            #  uint64_t len
            fmt = 'Q'
            length = calcsize(fmt)
            (dummy_len,) = unpack(fmt, buffer[offset:offset+length])
            logging.debug("dummy_len:{}".format(dummy_len))
            offset += length
            if dummy_len > 0:
                offset += 4 * dummy_len
        return (offset, XGBTree(tree_param, nodes))


class XGBModel(object):

    def __init__(self, num_feature, num_output_group, base_score):
        self.num_feature = num_feature
        self.num_output_group = num_output_group
        self.global_bias = base_score

        self.trees = []
        self.random_forest_flag = False
        self.pred_transform = "sigmoid"
        self.sigmoid_alpha = 1.0

    @staticmethod
    def load(buffer):
        # >>> LearnerModelParam
        offset = 0
        length, mparam = LearnerModelParam.load(buffer[offset:])
        offset += length
        # <<< LearnerModelParam

        #  uint64_t len
        fmt = 'Q'
        length = calcsize(fmt)
        (name_obj_str_len,) = unpack(fmt, buffer[offset:offset+length])
        logging.debug("name_obj_str_len:{}".format(name_obj_str_len))
        offset += length

        # string name_obj
        name_obj = buffer[offset:offset +
                          name_obj_str_len].decode(encoding='ascii')
        logging.debug("name_obj:{}".format(name_obj))
        offset += name_obj_str_len

        #  uint64_t len
        fmt = 'Q'
        length = calcsize(fmt)
        (name_gbm_str_len,) = unpack(fmt, buffer[offset:offset+length])
        logging.debug("name_gbm_str_len:{}".format(name_gbm_str_len))
        offset += length

        # string name_gbm
        name_gbm = buffer[offset:offset +
                          name_gbm_str_len].decode(encoding='ascii')
        logging.debug("name_gbm:{}".format(name_gbm))
        offset += name_gbm_str_len

        # >>> GBTreeModelParam
        length, gbm_parm = GBTreeModelParam.load(buffer[offset:])
        offset += length
        # <<< GBTreeModelParam

        xgb_trees = []
        for i in range(gbm_parm.num_trees):
            # print("offset: {}".format(offset))
            print("tree: {},".format(i), end='')
            length, tree = XGBTree.load(buffer[offset:])
            xgb_trees.append(tree)
            # print("length: {}".format(length))
            offset += length

        model = XGBModel(gbm_parm.num_feature,
                         gbm_parm.num_output_group, mparam.base_score)
        model.trees = xgb_trees
        return model

    def predictValueInst(self, feat):
        preds = []
        for tree in self.trees:
            nid = tree.get_leaf_index(feat)
            value = tree.get_value(nid)
            preds.append(value)
        return preds

    def predictLeafInst(self, indices, values):
        feat = FVec(self.num_feature)
        feat.fill(indices, values)
        preds = []
        for tree in self.trees:
            nid = tree.get_leaf_index(feat)
            preds.append(nid)
        return preds


class FVec(object):
    def __init__(self, size):
        self.vec = {}

    def fill(self, indices, values):
        for i in range(len(indices)):
            self.vec[indices[i]] = values[i]

    def drop(self, indices):
        for i in indices:
            del self.vec[indices[i]]

    def fvalue(self, index):
        return self.vec[index]

    def is_missing(self, index):
        return not (index in self.vec)
