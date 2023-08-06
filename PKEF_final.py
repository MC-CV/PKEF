import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
from DataHandler import negSamp, transpose, DataHandler, transToLsts
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle
import scipy.sparse as sp
from print_hook import PrintHook
import datetime
from time import time
import random


class Recommender:
    def __init__(self, sess, handler):
        self.sess = sess
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        self.metrics = dict()
        self.behEmbeds = NNs.defineParam('behEmbeds', [args.behNum, args.latdim // 2])
        if args.data == 'beibei':
            mets = ['Loss', 'preLoss', 'HR', 'NDCG', 'HR45', 'NDCG45', 'HR50', 'NDCG50', 'HR55', 'NDCG55', 'HR60',
                    'NDCG60', 'HR65', 'NDCG65', 'HR100', 'NDCG100']
        else:
            mets = ['Loss', 'preLoss', 'HR', 'NDCG', 'HR20', 'NDCG20', 'HR25', 'NDCG25', 'HR30', 'NDCG30', 'HR35',
                    'NDCG35', 'HR100', 'NDCG100']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()


    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            init = tf.global_variables_initializer()
            self.sess.run(init)
            log('Variables Inited')
        train_time = 0
        test_time = 0
        for ep in range(stloc, args.epoch):
            test = (ep % args.tstEpoch == 0)
            t0 = time()
            reses = self.trainEpoch()
            t1 = time()
            train_time += t1 - t0
            print('Train_time', t1 - t0, 'Total_time', train_time)
            log(self.makePrint('Train', ep, reses, test))
            if test:
                t2 = time()
                reses = self.testEpoch()
                t3 = time()
                test_time += t3 - t2
                print('Test_time', t3 - t2, 'Total_time', test_time)
                log(self.makePrint('Test', ep, reses, test))
            # if ep % args.tstEpoch == 0:
            #     self.saveHistory()
            print()
        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))

        #ADD
        log_dir = 'log/' + args.data + '/' + os.path.basename(__file__)

        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        # log_file = open(log_dir + '/log' + str(datetime.datetime.now()), 'w')
        log_file = open(
            log_dir + '/alllog', 'a')
        log_file.write("gnn_layer: " + str(args.gnn_layer) + " coefficient: " + str(args.coefficient) + "\n")
        log_file.write(self.makePrint('Test', args.epoch, reses, True))
        log_file.write("\n")
        log_file.write("\n")

        # self.saveHistory()

    def defineModel_parallel(self, allEmbed):
        gnn_layer = eval(args.gnn_layer)
        self.ulat_single = [0] * (args.behNum)
        self.ilat_single = [0] * (args.behNum)
        self.all_beh_embeddings = [] # [beh,layer+1]
        for beh in range(args.behNum):
            ego_embeddings = allEmbed
            all_embeddings = [ego_embeddings]
            for index in range(gnn_layer[beh]):
                symm_embeddings = tf.sparse_tensor_dense_matmul(self.adjs[beh], all_embeddings[-1])
                if args.encoder == 'lightgcn':
                    lightgcn_embeddings = symm_embeddings + all_embeddings[-1]
                    all_embeddings.append(lightgcn_embeddings)
            self.all_beh_embeddings.append(all_embeddings)
            all_embeddings = tf.add_n(all_embeddings)
            self.ulat_single[beh], self.ilat_single[beh] = tf.split(all_embeddings, [args.user, args.item], 0)


    def defineModel_cascade(self, allEmbed):
        gnn_layer = eval(args.gnn_layer)
        self.ulat = [0] * (args.behNum)
        self.ilat = [0] * (args.behNum)
        ego_embeddings = allEmbed
        for beh in range(args.behNum):
            all_embeddings = [ego_embeddings]
            for index in range(gnn_layer[beh]):
                symm_embeddings = tf.sparse_tensor_dense_matmul(self.adjs[beh], all_embeddings[-1])
                if args.encoder == 'lightgcn':
                    lightgcn_embeddings = symm_embeddings
                    lightgcn_embeddings = lightgcn_embeddings + all_embeddings[-1] 
                    lightgcn_sha_embeddings, _ = self.denoise(self.all_beh_embeddings[beh][index + 1], symm_embeddings)
                    lightgcn_embeddings = lightgcn_embeddings + lightgcn_sha_embeddings
                    all_embeddings.append(lightgcn_embeddings)
            ego_embeddings = all_embeddings[-1] + ego_embeddings
            all_embeddings = tf.add_n(all_embeddings)
            self.ulat[beh], self.ilat[beh] = tf.split(all_embeddings, [args.user, args.item], 0)
        self.ulat_merge, self.ilat_merge = tf.add_n(self.ulat), tf.add_n(self.ilat)

    def parallel_predict(self, src):
        uids = self.uids[src]
        iids = self.iids[src]

        tmp_emb_user = self.ulat_single[src]
        tmp_emb_item = self.ilat_single[src]

        src_ulat = tf.nn.embedding_lookup(tmp_emb_user, uids)
        src_ilat = tf.nn.embedding_lookup(tmp_emb_item, iids)

        exper_info = src_ulat * src_ilat

        preds = tf.squeeze(tf.reduce_sum(exper_info, axis=-1))

        return preds * args.mult

    def pme_predict(self, src):

        uids = self.uids[src]
        iids = self.iids[src]

        uids2 = self.uids2[src]
        iids2 = self.iids2[src]

        iids_other = self.iids_other[src]

        tmp_emb_user = self.ulat[src]
        tmp_emb_item = self.ilat[src]

        src_ulat = tf.nn.embedding_lookup(tmp_emb_user, uids)
        src_ilat = tf.nn.embedding_lookup(tmp_emb_item, iids)


        metalat111 = FC(tf.concat([src_ulat, src_ilat], axis=-1), args.behNum, reg=True, useBias=True,
                        activation='softmax', name='gate111', reuse=True)
        w1 = tf.reshape(metalat111, [-1, args.behNum, 1])


        exper_info = [src_ulat * src_ilat]
        uni_info = []
        self.tmp_uni_user = []
        self.tmp_uni_item = []
        for index in range(args.behNum):

            if index != src:
                tmp_sha_user, tmp_uni_user = self.denoise_pme(self.ulat[index], tmp_emb_user)
                tmp_sha_item, tmp_uni_item = self.denoise_pme(self.ilat[index], tmp_emb_item)

                if isinstance(iids_other[index], int) == True:
                    exper_info.append(
                        tf.nn.embedding_lookup(tmp_sha_user, uids) * tf.nn.embedding_lookup(tmp_sha_item, iids))
                else:
                    exper_info.append(
                        tf.nn.embedding_lookup(tmp_sha_user, uids) * tf.nn.embedding_lookup(tmp_sha_item, iids))
                    uni_info.append(
                        tf.nn.embedding_lookup(tmp_uni_user, uids2[index]) * tf.nn.embedding_lookup(tmp_uni_item,
                                                                                                    iids_other[index]))

        predEmbed = tf.stack(exper_info, axis=2)

        pme_out = tf.reshape(predEmbed @ w1, [-1, args.latdim // 2])


        preds = tf.squeeze(tf.reduce_sum(pme_out, axis=-1))
        preds_uni = []

        for i in uni_info:
            preds_uni.append(tf.squeeze(tf.reduce_sum(i, axis=-1)) * args.mult)

        return preds * args.mult, preds_uni


    def create_multiple_adj_mat(self, adj_mat):
        def left_adj_single(adj):

            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)


            norm_adj = d_mat_inv.dot(adj)
            print('generate left_adj_single adjacency matrix.')
            return norm_adj.tocoo()

        def right_adj_single(adj):
            rowsum = np.array(adj.sum(0))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = adj.dot(d_mat_inv)
            print('generate right_adj_single adjacency matrix.')
            return norm_adj.tocoo()

        def symm_adj_single(adj_mat):
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)


            rowsum = np.array(adj_mat.sum(0))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv_trans = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv_trans)
            print('generate symm_adj_single adjacency matrix.')
            return norm_adj.tocoo()

        left_adj_mat = left_adj_single(adj_mat)
        right_adj_mat = right_adj_single(adj_mat)
        symm_adj_mat = symm_adj_single(adj_mat)

        return left_adj_mat.tocsr(), right_adj_mat.tocsr(), symm_adj_mat.tocsr()

    def denoise(self, origin_emb, target_emb):
        res_array = tf.expand_dims(tf.reduce_sum(tf.multiply(origin_emb, target_emb), axis=1), -1) * target_emb
        norm_num = tf.norm(target_emb, axis=1) * tf.norm(target_emb, axis=1) + 1e-12
        clear_emb = res_array / tf.expand_dims(norm_num, -1)
        noise_emb = origin_emb - clear_emb
        if False:
            a = tf.cast(tf.reduce_sum(tf.multiply(origin_emb, target_emb), axis=1) >= 0, tf.float32)
            clear_emb *= tf.expand_dims(a, -1)
        return clear_emb * 0.3, noise_emb * 0.3

    def denoise_pme(self, origin_emb, target_emb):
        res_array = tf.expand_dims(tf.reduce_sum(tf.multiply(origin_emb, target_emb), axis=1), -1) * target_emb
        norm_num = tf.norm(target_emb, axis=1) * tf.norm(target_emb, axis=1) + 1e-12
        clear_emb = res_array / tf.expand_dims(norm_num, -1)
        noise_emb = origin_emb - clear_emb
        if False:
            a = tf.cast(tf.reduce_sum(tf.multiply(origin_emb, target_emb), axis=1) >= 0, tf.float32)
            clear_emb *= tf.expand_dims(a, -1)
        return clear_emb * 0.01, noise_emb * 0.01

    def prepareModel(self):
        self.actFunc = 'leakyRelu'
        self.adjs = []
        self.uids, self.iids = [], []
        self.uids2, self.iids2 = [], []
        self.iids_other = []
        self.left_trnMats, self.right_trnMats, self.symm_trnMats, self.none_trnMats = [], [], [], []

        for i in range(args.behNum):

            R = self.handler.trnMats[i].tolil()

            coomat = sp.coo_matrix(R)
            coomat_t = sp.coo_matrix(R.T)
            row = np.concatenate([coomat.row, coomat_t.row + R.shape[0]])
            col = np.concatenate([R.shape[0] + coomat.col, coomat_t.col])
            data = np.concatenate([coomat.data.astype(np.float32), coomat_t.data.astype(np.float32)])

            adj_mat = sp.coo_matrix((data, (row, col)), shape=(args.user + args.item, args.user + args.item))


            left_trn, right_trn, symm_trn = self.create_multiple_adj_mat(adj_mat)
            self.left_trnMats.append(left_trn)
            self.right_trnMats.append(right_trn)
            self.symm_trnMats.append(symm_trn)
            self.none_trnMats.append(adj_mat.tocsr())
        if args.normalization == "left":
            self.final_trnMats = self.left_trnMats
        elif args.normalization == "right":
            self.final_trnMats = self.right_trnMats
        elif args.normalization == "symm":
            self.final_trnMats = self.symm_trnMats
        elif args.normalization == 'none':
            self.final_trnMats = self.none_trnMats

        for i in range(args.behNum):
            adj = self.final_trnMats[i]
            idx, data, shape = transToLsts(adj, norm=False)
            self.adjs.append(tf.sparse.SparseTensor(idx, data, shape))

            self.uids.append(tf.placeholder(name='uids' + str(i), dtype=tf.int32, shape=[None]))
            self.iids.append(tf.placeholder(name='iids' + str(i), dtype=tf.int32, shape=[None]))
            tmp = []
            tmp1 = []
            tmp2 = []
            for j in range(args.behNum):
                if i != j and isinstance(self.handler.trnMats_uni_final[i][j], int) == False:
                    tmp.append(
                        tf.placeholder(name='iids_other' + str(i) + 'without' + str(j), dtype=tf.int32, shape=[None]))
                    tmp1.append(
                        tf.placeholder(name='uids2' + str(i) + 'without' + str(j), dtype=tf.int32, shape=[None]))
                    tmp2.append(
                        tf.placeholder(name='iids2' + str(i) + 'without' + str(j), dtype=tf.int32, shape=[None]))
                else:
                    tmp.append(0)
                    tmp1.append(0)
                    tmp2.append(0)
            self.iids_other.append(tmp)
            self.uids2.append(tmp1)
            self.iids2.append(tmp2)

        uEmbed0 = NNs.defineParam('uEmbed0', [args.user, args.latdim // 2], reg=True)
        iEmbed0 = NNs.defineParam('iEmbed0', [args.item, args.latdim // 2], reg=True)
        allEmbed = tf.concat([uEmbed0, iEmbed0], axis=0)

        self.defineModel_parallel(allEmbed)
        self.defineModel_cascade(allEmbed)

        self.preLoss = 0
        self.coefficient = eval(args.coefficient)
        for src in range(args.behNum):
            if args.decoder == 'pme':
                preds, preds_uni = self.pme_predict(src)
                tmp = 0
                for j in range(args.behNum):
                    if isinstance(self.uids2[src][j], int) == False:
                        sampNum = tf.shape(self.uids2[src][j])[0] // 2
                        posPred = tf.slice(preds_uni[tmp], [0], [sampNum])
                        negPred = tf.slice(preds_uni[tmp], [sampNum], [-1])
                        tmp += 1
                        self.preLoss += self.coefficient[src] * (tf.reduce_mean(tf.nn.softplus(-(posPred - negPred))) * 3)

            sampNum = tf.shape(self.uids[src])[0] // 2
            posPred = tf.slice(preds, [0], [sampNum])
            negPred = tf.slice(preds, [sampNum], [-1])


            preds_single = self.parallel_predict(src)
            posPred_single = tf.slice(preds_single, [0], [sampNum])
            negPred_single = tf.slice(preds_single, [sampNum], [-1])
            self.preLoss += self.coefficient[src] * (tf.reduce_mean(tf.nn.softplus(-(posPred_single - negPred_single))) * 3)

            self.preLoss += self.coefficient[src] * (tf.reduce_mean(tf.nn.softplus(-(posPred - negPred))) * 3)
            if src == args.behNum - 1:
                self.targetPreds = preds
        self.regLoss = args.reg * Regularize()
        self.loss = self.preLoss + self.regLoss

        globalStep = tf.Variable(0, trainable=False)
        learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

    def sampleTrainBatch(self, batIds, labelMat, labelMat_uni):
        temLabel = labelMat[batIds].toarray()
        temLabel_uni = labelMat_uni[batIds].toarray()
        batch = len(batIds)
        temlen = batch * 2 * args.sampNum
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        iLocs_uni = [None] * temlen
        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i] != 0), [-1])
            posset_uni = np.reshape(np.argwhere(temLabel_uni[i] != 0), [-1])
            sampNum = min(args.sampNum, len(posset), len(posset_uni))
            if sampNum == 0:
                poslocs = [np.random.choice(args.item)]
                neglocs = [poslocs[0]]
                poslocs_uni = [np.random.choice(args.item)]
                neglocs_uni = [poslocs_uni[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                poslocs_uni = np.random.choice(posset_uni, sampNum)
                neglocs = negSamp(temLabel[i], sampNum, args.item)
                neglocs_uni = negSamp(temLabel_uni[i], sampNum, args.item)
            for j in range(sampNum):
                posloc = poslocs[j]
                posloc_uni = poslocs_uni[j]
                negloc = neglocs[j]
                negloc_uni = neglocs_uni[j]
                uLocs[cur] = uLocs[cur + temlen // 2] = batIds[i]
                iLocs[cur] = posloc
                iLocs[cur + temlen // 2] = negloc
                iLocs_uni[cur] = posloc_uni
                iLocs_uni[cur + temlen // 2] = negloc_uni
                cur += 1
        uLocs = uLocs[:cur] + uLocs[temlen // 2: temlen // 2 + cur]
        iLocs = iLocs[:cur] + iLocs[temlen // 2: temlen // 2 + cur]
        iLocs_uni = iLocs_uni[:cur] + iLocs_uni[temlen // 2: temlen // 2 + cur]
        return uLocs, iLocs, iLocs_uni

    def sampleTrainBatch_ori(self, batIds, labelMat):
        temLabel = labelMat[batIds].toarray()
        batch = len(batIds)
        temlen = batch * 2 * args.sampNum
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i] != 0), [-1])
            sampNum = min(args.sampNum, len(posset))
            if sampNum == 0:
                poslocs = [np.random.choice(args.item)]
                neglocs = [poslocs[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                neglocs = negSamp(temLabel[i], sampNum, args.item)
            for j in range(sampNum):
                posloc = poslocs[j]
                negloc = neglocs[j]
                uLocs[cur] = uLocs[cur + temlen // 2] = batIds[i]
                iLocs[cur] = posloc
                iLocs[cur + temlen // 2] = negloc
                cur += 1
        uLocs = uLocs[:cur] + uLocs[temlen // 2: temlen // 2 + cur]
        iLocs = iLocs[:cur] + iLocs[temlen // 2: temlen // 2 + cur]
        return uLocs, iLocs

    def trainEpoch(self):
        num = args.user
        sfIds = np.random.permutation(num)[:args.trnNum]
        epochLoss, epochPreLoss = [0] * 2
        num = len(sfIds)
        steps = int(np.ceil(num / args.batch))
        for i in range(steps):
            st = i * args.batch
            ed = min((i + 1) * args.batch, num)
            batIds = sfIds[st: ed]

            target = [self.optimizer, self.preLoss, self.regLoss, self.loss]
            feed_dict = {}
            for beh in range(args.behNum):
                uLocs, iLocs = self.sampleTrainBatch_ori(batIds, self.handler.trnMats[beh])
                trnmat_uni = self.handler.trnMats_uni_final[beh]
                for beh_uni in range(args.behNum):
                    if isinstance(trnmat_uni[beh_uni], int) == True:
                        feed_dict[self.uids[beh]] = uLocs
                        feed_dict[self.iids[beh]] = iLocs
                    else:
                        uLocs2, iLocs2, iLocs_uni = self.sampleTrainBatch(batIds, self.handler.trnMats[beh],
                                                                          trnmat_uni[beh_uni])
                        feed_dict[self.uids[beh]] = uLocs
                        feed_dict[self.iids[beh]] = iLocs
                        feed_dict[self.uids2[beh][beh_uni]] = uLocs2
                        feed_dict[self.iids2[beh][beh_uni]] = iLocs2
                        feed_dict[self.iids_other[beh][beh_uni]] = iLocs_uni

            res = self.sess.run(target, feed_dict=feed_dict,
                                options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            preLoss, regLoss, loss = res[1:]

            epochLoss += loss
            epochPreLoss += preLoss
        ret = dict()
        ret['Loss'] = epochLoss / steps
        ret['preLoss'] = epochPreLoss / steps
        return ret

    def sampleTestBatch_allitem(self, batIds, labelMat):
        batch = len(batIds)
        temTst = self.handler.tstInt[batIds]
        temLabel = labelMat[batIds].toarray()
        temlen = batch * 100
        uLocs = []
        iLocs = []
        tstLocs = [None] * batch
        cur = 0
        for i in range(batch):
            posloc = temTst[i]
            negset = np.reshape(np.argwhere(temLabel[i] == 0), [-1])
            if isinstance(posloc, int) == True or type(posloc) in [np.int64]:
                posloc = int(posloc)
                locset = np.concatenate((negset, np.array([posloc])))
            else:
                locset = np.concatenate((negset, np.array(posloc)))

            tstLocs[i] = locset
            uLocs += [batIds[i]] * len(locset)
            iLocs += list(locset)
        return uLocs, iLocs, temTst, tstLocs

    def testEpoch(self):
        epochHit, epochNdcg = [0] * 2
        ids = self.handler.tstUsrs
        num = len(ids)
        tstBat = args.batch
        steps = int(np.ceil(num / tstBat))
        for i in range(steps):
            st = i * tstBat
            ed = min((i + 1) * tstBat, num)
            batIds = ids[st: ed]
            feed_dict = {}
            uLocs, iLocs, temTst, tstLocs = self.sampleTestBatch_allitem(batIds, self.handler.trnMats[-1])
            feed_dict[self.uids[-1]] = uLocs
            feed_dict[self.iids[-1]] = iLocs
            preds = self.sess.run(self.targetPreds, feed_dict=feed_dict,
                                  options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            hit, ndcg = self.calcRes_allitem(preds, temTst, tstLocs)
            epochHit += hit
            epochNdcg += ndcg

        ret = dict()
        ret['HR'] = epochHit / num
        ret['NDCG'] = epochNdcg / num
        return ret

    def calcRes_allitem(self, preds, temTst, tstLocs):
        hit = 0
        ndcg = 0
        batch = len(tstLocs)
        st = 0
        for j in range(batch):
            u_item_num = len(tstLocs[j]) + st
            cur_pred = preds[st:u_item_num]
            st = u_item_num
            predvals = list(zip(cur_pred, tstLocs[j]))
            predvals.sort(key=lambda x: x[0], reverse=True)
            shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
            if isinstance(temTst[j], int) == True or type(temTst[j]) in [np.int64]:
                temTst[j] = int(temTst[j])
                if temTst[j] in shoot:
                    hit += 1
                    ndcg += np.reciprocal(np.log2(shoot.index(temTst[j]) + 2))
            else:
                for eachTst in temTst[j]:
                    if eachTst in shoot:
                        hit += 1
                        ndcg += np.reciprocal(np.log2(shoot.index(eachTst) + 2))
        return hit, ndcg

    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        saver = tf.train.Saver()
        saver.save(self.sess, 'Models/' + args.save_path)
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        saver = tf.train.Saver()
        saver.restore(sess, 'Models/' + args.load_model)
        with open('History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')


if __name__ == '__main__':

    random.seed(42)
    tf.set_random_seed(42)
    np.random.seed(42)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    log_dir = 'log/' + args.data + '/' + os.path.basename(__file__)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    log_file = open(log_dir + '/log' + str(datetime.datetime.now()), 'w')


    def my_hook_out(text):
        log_file.write(text)
        log_file.flush()
        return 1, 0, text


    ph_out = PrintHook()
    ph_out.Start(my_hook_out)

    print("Use gpu id:", args.gpu_id)
    for arg in vars(args):
        print(arg + '=' + str(getattr(args, arg)))

    logger.saveDefault = True
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    with tf.Session(config=config) as sess:
        recom = Recommender(sess, handler)
        recom.run()
