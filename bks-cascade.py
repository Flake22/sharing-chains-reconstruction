import os
import numpy as np
import argparse

from src.results import *

REJECT_LABEL = 'REJ'

# Main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of several features and configurations')
    parser.add_argument('-db', required=False, default='controlled', help='''one of datasets: [controlled, uncontrolled, iplab, iplab_irene, isima, ...]. Default = controlled''')
    parser.add_argument('-db_config', required=False, default=3, type=int, help='integer: how to select and label images, default = 3')
    parser.add_argument('-db_test', required=False, default='controlled', help='''one of datasets: [controlled, uncontrolled, iplab, iplab_irene, isima, ...]. Default = controlled''')
    parser.add_argument('-features', required=False, default=['HEADER+DCT+META'], nargs='+', help='one or more of [DCT, META, HEADER, DCT+META, HEADER+META, HEADER+DCT+META]. Default = HEADER+DCT+META')

    args = parser.parse_args()

    db_train = args.db
    db_test = args.db_test
    db_config = args.db_config
    methods_fusion = args.features

    if not os.path.isdir('results'):
        os.makedirs('results', exist_ok=True)

    # labelling
    classes = []
    for i in range(db_config):
        classes.append(get_labels(db_test, i+1))

    # Get train features and label
    DCT_test, META_test, Header_test, labels_test = read_features(db_test, db_config, 'test')
    
    # Setup name strings for output results
    fnames = []
    basename = 'BKS-waterfall-v3-conf'+str(db_config)
    for i in range(db_config):
        fname = basename+'-step'+str(i)

        with open(os.path.join('results', 'imgs-'+fname+'.tex'), 'w+') as fout:           
            fout.write('\\begin{figure}[htbp!]\n\\centering\n')

        fnames.append(fname)
    
    # Get predictors
    predictors, maps = read_or_generate_predictors(db_train, classes, db_config, methods_fusion)

    # Get fusers
    combined_fusers = [m for m in methods_fusion if '+' in m]
    fusers = read_or_get_fusers(db_train, db_config, predictors, classes, combined_fusers, maps)

    # Test all fusions 
    for m_fusion in methods_fusion:
        if m_fusion in combined_fusers:
       
            # Initialize confusion matrices and rejection counters
            cm = []
            rejections = []
            for i in range(db_config):
                N = len(classes[i])
                cm.append(np.zeros((N,N)))
                rejections.append(0)

            # Get D1 (first-step) predictions
            for n, r in enumerate(labels_test):
                
                predictions = []
                for m in m_fusion.split('+'):    
                    features = get_features(DCT_test, META_test, Header_test, m)
                    tmp_p = int(predictors['D1-'+m].predict([features[n]]))
                    tmp_p = maps['D1'][tmp_p]
                    predictions.append(tmp_p)
                
                p1, _ = fusers['D1'][m_fusion].fuse(predictions)
                p1 = p1[0]
                
                if not p1 == REJECT_LABEL:
        
                    try:
                        r1 = r.split('-')[-1]
                    except:
                        r1 = r

                    # Update confusion matrix D1
                    cm[0][classes[0].index(r1)][classes[0].index(p1)] += 1
                else:
                    rejections[0] += 1/len(labels_test)

                # Get D2 (second-step) predictions
                if db_config>1 and not p1 == REJECT_LABEL:
                    det = 'D2-'+p1
                    
                    predictions = []
                    for m in m_fusion.split('+'):    
                        features = get_features(DCT_test, META_test, Header_test, m)
                        tmp_p = int(predictors[det+'-'+m].predict([features[n]]))
                        tmp_p = maps[det][tmp_p]
                        predictions.append(tmp_p)
                        
                    p2, _ = fusers[det][m_fusion].fuse(predictions)
                    p2 = p2[0]
                    
                    if not p2 == REJECT_LABEL:
        
                        if r.count('-')==2:
                            r2 = r[3:]
                        else:
                            r2 = r
                            
                        # Update confusion matrix D2
                        cm[1][classes[1].index(r2)][classes[1].index(p2)] += 1
                    else:
                        rejections[1] += 1/len(labels_test)

                    #"""
                    # Get D3 predictions
                    if db_config==3 and not p2 == REJECT_LABEL:
                        if '-' not in p2:
                            cm[2][classes[2].index(r)][classes[2].index(p2)] += 1
                        else:
                            det = 'D3-'+p2
                            
                            predictions = []
                            for m in m_fusion.split('+'):    
                                features = get_features(DCT_test, META_test, Header_test, m)
                                tmp_p = int(predictors[det+'-'+m].predict([features[n]]))
                                tmp_p = maps[det][tmp_p]
                                predictions.append(tmp_p)
                                
                            p3, _ = fusers[det][m_fusion].fuse(predictions)
                            p3 = p3[0]
                            
                            if not p3 == REJECT_LABEL:
                                
                                # Update confusion matrix D3
                                cm[2][classes[2].index(r)][classes[2].index(p3)] += 1
                            else:
                                rejections[2] += 1/len(labels_test)
                                
                    #"""

            for i in range(db_config):
                save_waterfall(len(classes[i]), cm[i], classes[i],
                               fnames[i], m_fusion, rejections[i])
        else:
            cm = []
            for i in range(db_config):
                N = len(classes[i])
                cm.append(np.zeros((N,N)))

            # get D1 predictions
            features = get_features(DCT_test, META_test, Header_test, m_fusion)

            for f, r in zip(features, labels_test):
                p = predictors['D1-'+m_fusion].predict([f])[0]
                p1 = maps['D1'][p]

                
                try:
                    r1 = r.split('-')[-1]
                except:
                    r1 = r

                cm[0][classes[0].index(r1)][classes[0].index(p1)]+=1


                # get D2 predictions
                if db_config>1:
                    det = 'D2-'+p1
                    p = predictors[det+'-'+m_fusion].predict([f])[0]
                    p2 = maps[det][p]

                    if r.count('-')==2:
                        r2 = r[3:]
                    else:
                        r2 = r
                    
                    cm[1][classes[1].index(r2)][classes[1].index(p2)]+=1

                    #get D3 predictions
                    if db_config==3:
                        if '-' not in p2:
                            cm[2][classes[2].index(r)][classes[2].index(p2)]+=1
                        else:
                            det = 'D3-'+p2
                            p = predictors[det+'-'+m_fusion].predict([f])[0]
                            p3 = maps[det][p]
                            
                            cm[2][classes[2].index(r)][classes[2].index(p3)]+=1

            # save results
            for i in range(db_config):
                save_waterfall(len(classes[i]), cm[i], classes[i], fnames[i], m_fusion)

    for fname in fnames:
        with open(os.path.join('results', 'imgs-'+fname+'.tex'), 'a+') as fout:           
            fout.write('\\end{figure}')
