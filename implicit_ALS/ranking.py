
class Rank_list():
    def __init__(self, n_items):
        rank_list = {}
        for k in range(n_items+1):
            if k == 1:
                rank_list[k] = [0.0]
            elif k == 2:
                #rank_list[k] = [0.0, 100.0]
                rank_list[k] = [0.0, 50.0]
            elif k > 2:
                '''this convention considers rank percentage [0,x,x...,100%]'''
                #increment = 100.0/(k-1)
                #ranks = [0.0]
                #for i in range(1,k-1):
                #    ranks.append(increment*i)
                #ranks.append(100.0)
                # ------------------------------

                # ''this convention considers rank percentage [0,x,x...,]'''
                increment = 100.0/float(k)
                ranks = [0.0]
                for i in range(1,k):
                    ranks.append(increment*i)
                
                # ------------------------------
                rank_list[k]=ranks

        self.rank_list = rank_list

