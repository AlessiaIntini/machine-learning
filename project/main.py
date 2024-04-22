import GaussianDensity as gd
import ReadData as rd

if __name__ == '__main__':
    D, L = rd.load('trainData.txt')
    # ########################
    # ## FEATURES ANALYSIS ###
    # ########################
    # fa.plot_features(D,L)

    # ########################
    # ## DIM REDUCTION #######
    # ########################
    # dr.Dim_red(D, L)

    # Classification
    # (DTR, LTR), (DVAL, LVAL) = rd.split_db_2to1(D, L)
    # dr.classification(DTR, LTR, DVAL, LVAL)

    # ########################
    # ## Gausian Density ####
    # ########################
    gd.apply_ML(D, L)
