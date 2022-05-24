#fusion_module.py contains the FusionModule class, which contains the code for fusing
#categorical distributions according to Independent Opinion Pool as well as
#according to the fusion method porposed in Cruz 2018 for comparison.



import numpy as np



# receives input base distributions and fuses them
class FusionModule:


    def __init__(self, fusion_method):
        '''
        constructor for FusionModule

        :param fusion_method: fusion method to be used, 'IOP' or 'Cruz'
        '''
        self.fusion_method = fusion_method



    def get_fused_pruned_output(self, dists, q_i):
        '''
        fuses the input distributions and removes unavailable actions

        :param dists: input distributions to be fused
        :param q_i: row of the q function of current state
        :return: fused distribution
        '''
        dis_fused = self.get_fused_output(dists)


        # distribution is to be normalized afterwards
        if self.fusion_method == 'IOP':
            sum = 0
            for n, q in enumerate(q_i):
                if q == -np.Inf:
                    dis_fused[n] = -np.Inf
                else:
                    sum += dis_fused[n]
            dis_fused /= sum

        elif self.fusion_method == 'Cruz':
            for n, q in enumerate(q_i):
                if q == -np.Inf:
                    dis_fused[n] = -np.Inf


        return dis_fused



    def get_fused_output(self, dists):
        '''
        fuses distributions dists depending on the fusion method

        :param dists: input distributions to be fused
        :return: fused distribution
        '''
        if self.fusion_method == 'IOP':
            return self.__independent_opinion_pool_fusion(dists)
        elif self.fusion_method == 'Cruz':
            return self.__cruz_fusion(dists)
        else:
            print('This fusion method is not defined.')



    def __independent_opinion_pool_fusion(self, dists):
        '''
        uses Independent Opinion Pool to fuse distributions dists

        :param dists: input distributions to be fused
        :return: fused distribution
        '''
        out = np.zeros_like(dists[0])
        sum_up = 0
        for i in range(dists[0].shape[0]):
            if -np.Inf not in dists[:,i]:
                out[i] = np.prod(dists[:,i])
                sum_up += out[i]
            else:
                out[i] = -np.Inf
        out /= sum_up
        return out



    def __cruz_fusion(self, dists):
        '''
        uses method from Cruz to fuse two distributions dists

        :param dists: input distributions to be fused
        :return: fused distribution
        '''

        dis1 = dists[0]
        dis2 = dists[1]
        out = np.zeros_like(dis1)

        for i in range(dis1.shape[0]):
            if dis1[i] == -np.Inf or dis2[i] == -np.Inf:
                out[i] = -np.Inf

        label1 = np.argmax(dis1)
        conf1 = np.amax(dis1)

        label2 = np.argmax(dis2)
        conf2 = np.amax(dis2)

        if conf1 >= conf2:
            label = label1
        else:
            label = label2

        if label1 == label2:
            phi = conf1 + conf2
        else:
            phi = abs(conf1 - conf2)
        conf = np.log(1 + phi) / np.log(3)

        # return a distribution with only one entry
        out[label] = conf

        # return label, conf
        return out

