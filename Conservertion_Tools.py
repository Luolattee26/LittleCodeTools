import numpy as np
import pandas as pd
import time


class calc:

    def help(self):
        print('best_200_phastCon:get the best 200bp\'s phastCon result of each trans')
        print('get_trans_ID_from_gene:get transID from gene symbol')
        print('phloP_percent:get significant fraction of each trans')

    def __init__(self, trans_file=None, phastCon_file=None, phloP_file=None, target_list=None, whether_time=True):

        self.trans = trans_file
        self.phastCon = phastCon_file
        self.phloP = phloP_file
        self.target = target_list
        self.get_time = whether_time
        self.time = {'best_200_phastCon':0, 'get_trans_ID_from_gene':0, 
                    'normal_phyloP':0, 'phloP_percent':0}

        if ('ENST' in self.target[0]) == False:
            print('please run get_trans_ID_from_gene method first')
        else:
            print('use help method to get usage')

        header = ['chrom', 'chromStart', 'chromEnd', 'gene_ID',
                'trans_ID', 'Symbol', 'location', 'type']
        self.trans.columns = header[:len(self.trans.columns)]

    def best_200_phastCon(self, test=False, log=False, cores=1):
        count = 0
        best_200_list = []
        score_sum = []

        if ('ENST' in self.target[0]) == False:
            print('please run get_trans_ID_from_gene method first')
            return

        s = time.time()

        def process_trans(trans):
            if log == True:
                nonlocal count
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

            # calculate best 200bp
            scores = self.phastCon.values(chr_num, start, end)
            
            no_nan = []
            for i in scores:
                if np.isnan(i) == True:
                    no_nan.append(0)
                else:
                    no_nan.append(1)
                    
            no_nan_sums = np.cumsum(no_nan)
            no_nan_sums[200:] = no_nan_sums[200:] - no_nan_sums[:-200]
            no_nan_sums = [10 ** 10 if x == 0 else x for x in no_nan_sums]
            
            scores = np.nan_to_num(scores, nan=0)
            
            
            score_sums = np.cumsum(scores)
            score_sums[200:] = score_sums[200:] - score_sums[:-200]

            means = score_sums[199:] / no_nan_sums[199:]
            
            if means.size > 0:
                best_200_list.append(np.max(means))
                score_sum.append(np.sum(means))
            else:
                best_200_list.append('No data')
                score_sum.append('No data')

        # pool = Pool(processes=cores)
        # pool.map(process_trans, self.target)
        # pool.close()
        
        for i in self.target:
            process_trans(i)
        

        # to check the code, there should be different sum value between the trans
        if test == True:
            print('Next is the sum of each 200bp per trans, if the calculation is right, the number will be different form each other')
            print(score_sum)

        # output DataFrame
        df = pd.DataFrame({'transID': self.target, 'result': best_200_list})
        e = time.time()

        self.time['best_200_phastCon'] = e - s

        return df

    def get_trans_ID_from_gene(self):

        s = time.time()

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
        
        e = time.time()

        self.time['get_trans_ID_from_gene'] = e - s

        return self.target

    def phloP_percent(self, log=False, pvalue_cut=0.01):

        s = time.time()

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
            wig_score_sig = [x for x in wig_score_P if x < pvalue_cut]

            fraction.append(len(wig_score_sig) / len(wig_score))

        # output DataFrame
        df = pd.DataFrame({'transID': self.target, 'result': fraction})

        e = time.time()

        self.time['phloP_percent'] = e - s

        return df
    
    def normal_phyloP(self, log=False):

        s = time.time()

        if ('ENST' in self.target[0]) == False:
            print('please run get_trans_ID_from_gene method first')
            return
        
        mean_phyloP = []

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
            mean_value = np.mean(wig_score)

            mean_phyloP.append(mean_value)

            # print(sum(wig_score)/(start-end))

        # output DataFrame
        df = pd.DataFrame({'transID': self.target, 'result': mean_phyloP})

        e = time.time()

        self.time['normal_phyloP'] = e - s

        return df
    
    def time_recordor(self):
        if self.get_time:
            return self.time



if __name__ == '__main__':
    print('Running test \n target lncRNA: [TP53, H19]')

    import sys
    sys.path.append('/data/jxwang_data/WMDS_lncRNA/')

    import numpy as np
    import pyBigWig
    import pandas as pd

    lst1 = ['TP53', 'H19']
    all_trans_bed = pd.read_csv('/data/jxwang_data/WMDS_lncRNA/all.transcripts.bed', sep = '\t', header = None)
    Con_bw_file = pyBigWig.open('/data/jxwang_data/WMDS_lncRNA/hg38.phastCons100way.bw')
    P_bw_file = pyBigWig.open('/data/jxwang_data/WMDS_lncRNA/hg38.phyloP100way.bw')

    calc1 = calc(all_trans_bed, Con_bw_file, P_bw_file, lst1)

    calc1.get_trans_ID_from_gene()

    a = calc1.normal_phyloP()
    b = calc1.best_200_phastCon()
    c = calc1.phloP_percent()
    
    print('\n', 'phyloP mean', '\n', a, '\n', 'best 200 bp phastCon', b, '\n', 'phyloP percent', c, '\n', 'costing time', calc1.time_recordor())