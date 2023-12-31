import numpy as np
import pandas as pd
import time


class calc:

    def help(self):
        print('best_200_phastCon:get the best 200bp\'s phastCon result of each trans')
        print('get_trans_ID_from_gene:get transID from gene symbol')
        print('phloP_percent:get significant fraction of each trans')

    def __init__(self, trans_file=None, phastCon_file=None, phloP_file=None, target_list=None):

        self.trans = trans_file
        self.phastCon = phastCon_file
        self.phloP = phloP_file
        self.target = target_list

        if ('ENST' in self.target[0]) == False:
            print('please run get_trans_ID_from_gene method first')
        else:
            print('use help method to get usage')

        header = ['chrom', 'chromStart', 'chromEnd', 'gene_ID',
                  'trans_ID', 'Symbol', 'location', 'type']
        self.trans.columns = header[:len(self.trans.columns)]

    def best_200_phastCon(self, test=False, log=False):
        count = 0
        best_200_list = []
        score_sum = []

        if ('ENST' in self.target[0]) == False:
            print('please run get_trans_ID_from_gene method first')
            return

        s = time.time()
        for trans in self.target:

            if log == True:
                count += 1
                print(trans, 'Percent {}%'.format(
                    count / len(self.target) * 100))

            # get chrom position for every trans
            single_trans = self.trans.loc[self.trans.loc[:,
                                                         'trans_ID'] == trans, ]
            # print(single_trans)
            chr_num = single_trans.iloc[0, 0]
            start = single_trans.iloc[0, 1]
            end = single_trans.iloc[0, 2]

            # score list
            single_trans_score_list = []

            # calculate best 200bp
            for index in range(start, end):
                if (end - index) > 200:
                    score = self.phastCon.values(chr_num, index, (index+200))
                    score_no_NAN = [x for x in score if np.isnan(x) == False]
                    score_mean = np.mean(score_no_NAN)
                    single_trans_score_list.append(score_mean)
                elif (end - index) == 200:
                    score = self.phastCon.values(chr_num, index, end)
                    score_no_NAN = [x for x in score if np.isnan(x) == False]
                    score_mean = np.mean(score_no_NAN)
                    single_trans_score_list.append(score_mean)
                else:
                    break

            single_trans_score_list = [
                x for x in single_trans_score_list if np.isnan(x) == False]
            if single_trans_score_list != []:
                best_200_list.append(max(single_trans_score_list))
                score_sum.append(sum(single_trans_score_list))
            else:
                best_200_list.append("No data")
                score_sum.append("No data")

        # to check the code, there should be different sum value between the trans
        if test == True:
            print('Next is the sum of each 200bp per trans, if the calculation is right, the number will be different form each other')
            print(score_sum)

        # output DataFrame
        df = pd.DataFrame({'transID': self.target, 'result': best_200_list})
        e = time.time()
        print('cost time {}s'.format(e - s))

        return df

    def get_trans_ID_from_gene(self):

        if ('ENST' in self.target[0]) == True:
            print('no need to translate')
            return

        transID_list = []

        if ('ENSG' in self.target[0]) == True:
            gene_ID_set = set(self.trans.loc[:, "gene_ID"])
            lst = []

            for gene in self.target:
                for i in gene_ID_set:
                    if gene in i:
                        lst.append(i)

            for gene in lst:
                gene_to_bed = self.trans.loc[self.trans.loc[:,
                                                            'gene_ID'] == gene, ]
                transID = list(gene_to_bed.loc[:, 'trans_ID'])
                transID_list += transID

            self.target = transID_list
            print('the translation is done')
        else:
            for gene in self.target:
                gene_to_bed = self.trans.loc[self.trans.loc[:,
                                                            'Symbol'] == gene, ]
                transID = list(gene_to_bed.loc[:, 'trans_ID'])
                transID_list += transID

            self.target = transID_list
            print('the translation is done')

        return self.target

    def phloP_percent(self, log=False):

        if ('ENST' in self.target[0]) == False:
            print('please run get_trans_ID_from_gene method first')
            return

        fraction = []
        for trans in self.target:

            if log == True:
                print(trans)

            # get chrom position for every trans
            single_trans = self.trans.loc[self.trans.loc[:,
                                                         'trans_ID'] == trans, ]
            # print(single_trans)
            chr_num = single_trans.iloc[0, 0]
            start = single_trans.iloc[0, 1]
            end = single_trans.iloc[0, 2]

            # calculate the fraction
            wig_score = self.phloP.values(chr_num, start, end)
            wig_score_P = [(10 ^ int(-x)) for x in wig_score if x > 0]
            wig_score_sig = [x for x in wig_score_P if x < 0.01]

            fraction.append(len(wig_score_sig) / len(wig_score))

        # output DataFrame
        df = pd.DataFrame({'transID': self.target, 'result': fraction})

        return df


#########################################################################################################################
# examples
#########################################################################################################################

# import sys
# sys.path.append('/data/jxwang_data/WMDS_lncRNA/')


# import numpy as np
# import pyBigWig
# import pandas as pd
# from calc_phastCon_and_phloP import calc


# lst1 = ['TP53', 'H19']
# all_trans_bed = pd.read_csv('/data/jxwang_data/WMDS_lncRNA/all.transcripts.bed', sep = '\t', header = None)
# Con_bw_file = pyBigWig.open('/data/jxwang_data/WMDS_lncRNA/hg38.phastCons100way.bw')
# P_bw_file = pyBigWig.open('/data/jxwang_data/WMDS_lncRNA/hg38.phyloP100way.bw')

# calc1 = calc(all_trans_bed, Con_bw_file, P_bw_file, lst1)

# calc1.get_trans_ID_from_gene()


# a = calc1.phloP_percent()
# b = calc1.best_200_phastCon(test=True, log=True)

# print('\n','phyloP fraction','\n',a,'\n','best200_phastCon','\n',b)
