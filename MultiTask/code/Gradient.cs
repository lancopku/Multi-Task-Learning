//
//  Online Multi-Task Learning Toolkit (OMT) v1.0
//
//  Copyright(C) Xu Sun <xusun@pku.edu.cn> http://klcl.pku.edu.cn/member/sunxu/index.htm
//

using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Collections;

namespace Program
{
    class gradient
    {
        protected Optimizer _optim;
        protected inference _inf;
        protected featureGenerator _fGene;

        public gradient(toolbox tb)
        {
            _optim = tb.Optim;
            _inf = tb.Inf;
            _fGene = tb.FGene;
        }

        //return the gradient of -log{P(y*|x,w)} as follows: E_{P(y|x)}(F(x,y)) - F(x,y*)
        virtual public double getGrad(List<double> vecGrad, model m, dataSeq x, baseHashSet<int> idSet)
        {
            if (idSet != null) idSet.Clear();
            int nTag = m.NTag;
            //compute beliefs
            belief bel = new belief(x.Count, nTag);
            belief belMasked = new belief(x.Count, nTag);
            _inf.getBeliefs(bel,m, x, false);
            _inf.getBeliefs(belMasked,m, x, true);
            double ZGold = belMasked.Z;
            double Z = bel.Z;

            List<featureTemp> fList;
            for (int i = 0; i < x.Count; i++)
            {
                fList = _fGene.getFeatureTemp(x, i);
                for (int j = 0; j < fList.Count; j++)
                {
                    featureTemp im = fList[j];
                    int id = im.id;
                    double v = im.val;
                    for (int s = 0; s < nTag; s++)
                    {
                        int f = _fGene.getNodeFeatID(id,s);
                        if (idSet != null) idSet.Add(f);
                        vecGrad[f] += bel.belState[i][s] * v;
                        vecGrad[f] -= belMasked.belState[i][s] * v;
                    }
                }
            }

            for (int i = 1; i < x.Count; i++) 
            {
                for (int s = 0; s < nTag; s++)
                {
                    for (int sPre = 0; sPre < nTag; sPre++)
                    {
                        int f = _fGene.getEdgeFeatID(sPre, s);
                        if (idSet != null) idSet.Add(f);
                        vecGrad[f] += bel.belEdge[i][sPre, s];
                        vecGrad[f] -= belMasked.belEdge[i][sPre, s];
                    }
                }
            }
            return Z - ZGold;
        }

        //the scalar version
        virtual public double getGrad(List<double> vecGrad, double scalar, model m, dataSeq x, baseHashSet<int> idSet)
        {
            idSet.Clear();
            int nbStates = m.NTag;
            //compute beliefs
            belief bel = new belief(x.Count, nbStates);
            belief belMasked = new belief(x.Count, nbStates);
            _inf.getBeliefs(bel,m, x, scalar, false);
            _inf.getBeliefs(belMasked,m, x, scalar, true);
            double ZGold = belMasked.Z;
            double Z = bel.Z;

            List<featureTemp> fList;
            for (int i = 0; i < x.Count; i++)
            {
                fList = _fGene.getFeatureTemp(x, i);
                for (int j = 0; j < fList.Count; j++)
                {
                    featureTemp im = fList[j];
                    int id = im.id;
                    double v = im.val;
                    for (int s = 0; s < nbStates; s++)
                    {
                        int f =_fGene.getNodeFeatID(id,s);
                        idSet.Add(f);
                        vecGrad[f] += bel.belState[i][s] * v;
                        vecGrad[f] -= belMasked.belState[i][s] * v;
                    }
                }
            }

            for (int i = 1; i < x.Count; i++) 
            {
                for (int s = 0; s < nbStates; s++)
                {
                    for (int sPre = 0; sPre < nbStates; sPre++)
                    {
                        int f = _fGene.getEdgeFeatID(sPre, s);
                        idSet.Add(f);
                        vecGrad[f] += bel.belEdge[i][sPre, s];
                        vecGrad[f] -= belMasked.belEdge[i][sPre, s];
                    }
                }
            }
            return Z - ZGold;
        }

        //return the grad of -log{P(y*|x,w)}
        public double getGrad_SGD(List<double> vecGrad, model m, dataSeq x, baseHashSet<int> idSet)
        {
            if (idSet != null) 
                idSet.Clear();

            if (x == null)
                return 0;

            return getGrad(vecGrad, m, x, idSet);
        }

        public double getGrad_SGD(List<double> vecGrad, double scalar, model m, dataSeq x, baseHashSet<int> idset)
        {
            return getGrad(vecGrad, scalar, m, x, idset);
        }

        //compute grad of: sum{-log{P(y*|x,w)} + R(w)}
        public double getGrad_BFGS(List<double> vecGrad, model m, dataSet X)
        {
            //-log(obj)
            double error = 0;
            int nbFeatures = _fGene.NCompleteFeature;

            //int i = 0;
            foreach (dataSeq im in X)
            {
                double err = 0;
                err = getGrad(vecGrad, m, im, null);
                error += err;
            }

            if (Global.reg != 0.0)
            {
                for (int f = 0; f < nbFeatures; f++)
                {
                    vecGrad[f] += m.W[f] / (Global.reg * Global.reg);
                }
            }
            if (Global.reg != 0.0)
            {
                List<double> tmpWeights = m.W;
                double sum = listTool.squareSum(tmpWeights);
                error += sum / (2.0 * Global.reg * Global.reg);
            }
            return error;
        }

    }
}