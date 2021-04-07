"""define Factorization-Machine based Neural Network Model"""
import math
import numpy as np
import tensorflow as tf
from src.base_model import BaseModel

__all__ = ["ExtremeDeepFMModel"]


class ExtremeDeepFMModel(BaseModel):
    """define Factorization-Machine based Neural Network Model"""

    def _build_graph(self, hparams):
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)
        self.layer_keeps = tf.placeholder(tf.float32)
        with tf.variable_scope("exDeepFm") as scope:
            with tf.variable_scope("embedding", initializer=self.initializer) as escope:
                self.embedding = tf.get_variable(name='embedding_layer',
                                                 shape=[hparams.FEATURE_COUNT, hparams.dim],
                                                 dtype=tf.float32)
                self.embed_params.append(self.embedding)
                embed_out, embed_layer_size = self._build_embedding(hparams)
            logit = self._build_linear(hparams)######linear part

            exFM_out, final_result, attention_out = self._build_extreme_FM(hparams, embed_out, res=False, direct=True, bias=False, reduce_D=False, f_dim=2)

            logit = tf.add(logit, exFM_out)
            logit = tf.add(logit, self._build_dnn(hparams, embed_out, embed_layer_size))######DNN part
            return logit, final_result, attention_out         ######linear+extremeFM+dnn

    def _build_embedding(self, hparams):
        fm_sparse_index = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                          self.iterator.dnn_feat_values,
                                          self.iterator.dnn_feat_shape)
        fm_sparse_weight = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                           self.iterator.dnn_feat_weights,
                                           self.iterator.dnn_feat_shape)
        w_fm_nn_input_orgin = tf.nn.embedding_lookup_sparse(self.embedding,
                                                            fm_sparse_index,
                                                            fm_sparse_weight,
                                                            combiner="sum")
        embedding = tf.reshape(w_fm_nn_input_orgin, [-1, hparams.dim * hparams.FIELD_COUNT])
        embedding_size = hparams.FIELD_COUNT * hparams.dim
        return embedding, embedding_size


    def _build_linear(self, hparams):
        with tf.variable_scope("linear_part", initializer=self.initializer) as scope:
            w_linear = tf.get_variable(name='w',
                                       shape=[hparams.FEATURE_COUNT, 1],
                                       dtype=tf.float32)
            b_linear = tf.get_variable(name='b',
                                       shape=[1],
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer())
            x = tf.SparseTensor(self.iterator.fm_feat_indices,
                                self.iterator.fm_feat_values,
                                self.iterator.fm_feat_shape)
            linear_output = tf.add(tf.sparse_tensor_dense_matmul(x, w_linear), b_linear)
            self.layer_params.append(w_linear)
            self.layer_params.append(b_linear)
            tf.summary.histogram("linear_part/w", w_linear)
            tf.summary.histogram("linear_part/b", b_linear)
            return linear_output
        
    ##################cross part##############################################################
    def _build_extreme_FM(self, hparams, nn_input, res=False, direct=True, bias=False, reduce_D=False, f_dim=2):
        hidden_nn_layers = []
        field_nums = []
        final_len = 0
        field_num = hparams.FIELD_COUNT
        nn_input = tf.reshape(nn_input, shape=[-1, int(field_num), hparams.dim])
        field_nums.append(int(field_num))
        hidden_nn_layers.append(nn_input)
        final_result = []
        split_tensor0 = tf.split(hidden_nn_layers[0], hparams.dim * [1], 2)
        with tf.variable_scope("exfm_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.cross_layer_sizes):
                split_tensor = tf.split(hidden_nn_layers[-1], hparams.dim * [1], 2)
                dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
                dot_result_o = tf.reshape(dot_result_m, shape=[hparams.dim, -1, field_nums[0]*field_nums[-1]])
                dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

                ###############################################################
                ######  reduce_D=False, without Dimension reduction  ##########
                
                filters = tf.get_variable(name="f_"+str(idx),
                                     shape=[1, field_nums[-1]*field_nums[0], layer_size],
                                     dtype=tf.float32)

                curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')
                ###############################################################
                
                ###############################################################
                ######  bias=False, without bias  #############################
                ###############################################################

                curr_out = self._activate(curr_out, hparams.cross_activation)

                curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

                ###############################################################
                ######  direct=False, without split  ##########################
                if direct:
                    hparams.logger.info("all direct connect")
                    direct_connect = curr_out
                    next_hidden = curr_out
                    final_len += layer_size
                    field_nums.append(int(layer_size))

                else:
                    hparams.logger.info("split connect")
                    if idx != len(hparams.cross_layer_sizes) - 1:
                        next_hidden, direct_connect = tf.split(curr_out, 2 * [int(layer_size / 2)], 1)
                        final_len += int(layer_size / 2)
                    else:
                        direct_connect = curr_out
                        next_hidden = 0
                        final_len += layer_size
                    field_nums.append(int(layer_size / 2))
                ###############################################################
                
                direct_connect = tf.nn.l2_normalize(direct_connect, [1,2])

                final_result.append(direct_connect)
                hidden_nn_layers.append(next_hidden)

                self.cross_params.append(filters)
                self.layer_params.append(filters)


            result = tf.concat(final_result, axis=1)
            result = tf.reduce_sum(result, -1)#######sum
            
            temp_result = tf.reshape(result, shape=[-1, len(hparams.cross_layer_sizes), layer_size])
            
            attention_W = tf.get_variable(name="attention_W",
                                          shape=[hparams.attdim, layer_size],
                                          dtype=tf.float32)
            attention_b = tf.get_variable(name="attention_b",
                                          shape=[1, hparams.attdim],
                                          dtype=tf.float32)
            attention_h = tf.get_variable(name="attention_h",
                                          shape=[hparams.attdim],
                                          dtype=tf.float32)
            
            attention_mul = tf.reshape(tf.matmul(tf.reshape(temp_result, shape=[-1, layer_size]), \
                    attention_W), shape=[-1, len(hparams.cross_layer_sizes), hparams.attdim])
            
            attention_relu = tf.reduce_sum(tf.multiply(attention_h, tf.nn.relu(attention_mul + \
                attention_b)), 2, keep_dims=True)
            
            attention_out = tf.nn.softmax(attention_relu)
            
            
            atten_mul_result = tf.multiply(attention_out, temp_result)
            atten_mul_result = tf.reshape(atten_mul_result, shape=[-1, len(hparams.cross_layer_sizes)*layer_size]) #expected [?, 600]
            ###################################################################
            #####  res=False, No Residual Network  ############################
            hparams.logger.info("no residual network")
            w_nn_output = tf.get_variable(name='w_nn_output',
                                          shape=[final_len, 1],
                                          dtype=tf.float32)
            b_nn_output = tf.get_variable(name='b_nn_output',
                                          shape=[1],
                                          dtype=tf.float32,
                                          initializer=tf.zeros_initializer())
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            #exFM_out = tf.nn.xw_plus_b(result, w_nn_output, b_nn_output)#matmul(result, w_nn_output) + b_nn_output
            exFM_out = tf.nn.xw_plus_b(atten_mul_result, w_nn_output, b_nn_output)
            ###################################################################

            return exFM_out, final_result, attention_out##########################Add
        

    def _build_extreme_FM_quick(self, hparams, nn_input):
        hidden_nn_layers = []
        field_nums = []
        final_len = 0
        field_num = hparams.FIELD_COUNT
        nn_input = tf.reshape(nn_input, shape=[-1, int(field_num), hparams.dim])
        field_nums.append(int(field_num))
        hidden_nn_layers.append(nn_input)
        final_result = []
        split_tensor0 = tf.split(hidden_nn_layers[0], hparams.dim * [1], 2)
        with tf.variable_scope("exfm_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.cross_layer_sizes):
                split_tensor = tf.split(hidden_nn_layers[-1], hparams.dim * [1], 2)
                dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
                dot_result_o = tf.reshape(dot_result_m, shape=[hparams.dim, -1, field_nums[0]*field_nums[-1]])
                dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

                filters = tf.get_variable(name="f_"+str(idx),
                                         shape=[1, field_nums[-1]*field_nums[0], layer_size],
                                         dtype=tf.float32)

                curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')


                curr_out = tf.transpose(curr_out, perm=[0, 2, 1])


                hparams.logger.info("split connect")
                if idx != len(hparams.cross_layer_sizes) - 1:
                    next_hidden, direct_connect = tf.split(curr_out, 2 * [int(layer_size / 2)], 1)
                    final_len += int(layer_size / 2)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
                    final_len += layer_size
                field_nums.append(int(layer_size / 2))

                final_result.append(direct_connect)
                hidden_nn_layers.append(next_hidden)

                self.cross_params.append(filters)

            result = tf.concat(final_result, axis=1)
            result = tf.reduce_sum(result, -1)

            hparams.logger.info("no residual network")
            w_nn_output = tf.get_variable(name='w_nn_output',
                                              shape=[final_len, 1],
                                              dtype=tf.float32)
            b_nn_output = tf.get_variable(name='b_nn_output',
                                              shape=[1],
                                              dtype=tf.float32,
                                              initializer=tf.zeros_initializer())
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            exFM_out = tf.nn.xw_plus_b(result, w_nn_output, b_nn_output)

            return exFM_out


    def _build_dnn(self, hparams, embed_out, embed_layer_size):
        """
        fm_sparse_index = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                          self.iterator.dnn_feat_values,
                                          self.iterator.dnn_feat_shape)
        fm_sparse_weight = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                           self.iterator.dnn_feat_weights,
                                           self.iterator.dnn_feat_shape)
        w_fm_nn_input_orgin = tf.nn.embedding_lookup_sparse(self.embedding,
                                                            fm_sparse_index,
                                                            fm_sparse_weight,
                                                            combiner="sum")
        w_fm_nn_input = tf.reshape(w_fm_nn_input_orgin, [-1, hparams.dim * hparams.FIELD_COUNT])
        last_layer_size = hparams.FIELD_COUNT * hparams.dim
        """
        w_fm_nn_input = embed_out
        last_layer_size = embed_layer_size
        layer_idx = 0
        hidden_nn_layers = []
        hidden_nn_layers.append(w_fm_nn_input)
        with tf.variable_scope("nn_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.layer_sizes):
                curr_w_nn_layer = tf.get_variable(name='w_nn_layer' + str(layer_idx),
                                                  shape=[last_layer_size, layer_size],
                                                  dtype=tf.float32)
                curr_b_nn_layer = tf.get_variable(name='b_nn_layer' + str(layer_idx),
                                                  shape=[layer_size],
                                                  dtype=tf.float32,
                                                  initializer=tf.zeros_initializer())
                tf.summary.histogram("nn_part/" + 'w_nn_layer' + str(layer_idx),
                                     curr_w_nn_layer)
                tf.summary.histogram("nn_part/" + 'b_nn_layer' + str(layer_idx),
                                     curr_b_nn_layer)
                curr_hidden_nn_layer = tf.nn.xw_plus_b(hidden_nn_layers[layer_idx],
                                                       curr_w_nn_layer,
                                                       curr_b_nn_layer)
                scope = "nn_part" + str(idx)
                activation = hparams.activation[idx]
                curr_hidden_nn_layer = self._active_layer(logit=curr_hidden_nn_layer,
                                                          scope=scope,
                                                          activation=activation,
                                                          layer_idx=idx)
                hidden_nn_layers.append(curr_hidden_nn_layer)
                layer_idx += 1
                last_layer_size = layer_size
                self.layer_params.append(curr_w_nn_layer)
                self.layer_params.append(curr_b_nn_layer)

            w_nn_output = tf.get_variable(name='w_nn_output',
                                          shape=[last_layer_size, 1],
                                          dtype=tf.float32)
            b_nn_output = tf.get_variable(name='b_nn_output',
                                          shape=[1],
                                          dtype=tf.float32,
                                          initializer=tf.zeros_initializer())
            tf.summary.histogram("nn_part/" + 'w_nn_output' + str(layer_idx),
                                 w_nn_output)
            tf.summary.histogram("nn_part/" + 'b_nn_output' + str(layer_idx),
                                 b_nn_output)
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            nn_output = tf.nn.xw_plus_b(hidden_nn_layers[-1], w_nn_output, b_nn_output)
            return nn_output
