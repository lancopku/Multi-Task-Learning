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
using System.Diagnostics;

namespace Program
{
    class optimSGD : Optimizer
    {
        double[,] _simiBiAry;
        List<dataSet> _newXList;

        public optimSGD(toolbox tb)
        {
            _model = tb.Model;
            _modelList = tb.ModelList;
            _X = tb.X;
            _XList = tb.XList;
            _inf = tb.Inf;
            _fGene = tb.FGene;
            _grad = tb.Grad;
     
            //init
            if (Global.runMode.Contains("mt"))
                initForMulti();
        }

        override public double optimize()
        {
            double error = 0;

            if (Global.optim.Contains("sgder"))
                error = sgd_exactReg();
            else error = sgd_lazyReg();
            
            return error;
        }

        override public double optimize_multi()
        {
            if(Global.runMode.Contains("fast"))
                return sgd_multi_fast();
            else
                return sgd_multi();
        }

        void initForMulti()
        {
            _simiBiAry = new double[Global.nTask, Global.nTask];
            for (int i = 0; i < Global.nTask; i++)
                _simiBiAry[i, i] = 1;

            if (_XList != null)
            {
                _newXList = new List<dataSet>();

                List<int> sizeList = new List<int>();
                int maxSize = 0;
                for (int i = 0; i < _XList.Count; i++)
                {
                    dataSet Xi = _XList[i];
                    int size = Xi.Count;
                    sizeList.Add(size);
                    if (maxSize < size)
                        maxSize = size;
                }

                for (int i = 0; i < Global.nTask; i++)
                {
                    dataSet X = new dataSet();
                    dataSet Xi = _XList[i];
                    foreach (dataSeq x in Xi)
                        X.Add(x);
                    //to make Xs in newXList have the same length
                    for (int k = sizeList[i]; k < maxSize; k++)
                        X.Add(null);
                    _newXList.Add(X);
                }
            }
        }

        public double sgd_lazyReg()
        {
            List<double> w = _model.W;
            int fsize = w.Count;
            int xsize = _X.Count;
            double[] ary = new double[fsize];
            List<double> grad = new List<double>(ary);

            List<int> ri = randomTool<int>.getShuffledIndexList(xsize);
            double error = 0;
            double r_k = 0;

            for (int t = 0; t < xsize; t++)
            {
                int ii = ri[t];
                dataSeq x = _X[ii];
                baseHashSet<int> fset = new baseHashSet<int>();
                double err = _grad.getGrad_SGD(grad, _model, x, fset);
                error += err;
                
                r_k = Global.rate0 * Math.Pow(Global.decayFactor, (double)Global.countWithIter / (double)xsize);
                if (Global.countWithIter % (xsize / 4) == 0)
                    Global.swLog.WriteLine("iter{0}    decay_rate={1}", Global.glbIter, r_k.ToString("e2"));

                foreach (int i in fset)
                {
                    w[i] -= r_k * grad[i];
                    //reset
                    grad[i] = 0;
                }
                Global.countWithIter++;
            }

            if (Global.reg != 0)
            {
                for (int i = 0; i < fsize; i++)
                {
                    double grad_i = w[i] / (Global.reg * Global.reg);
                    w[i] -= r_k * grad_i;
                }

                double sum = listTool.squareSum(w);
                error += sum / (2.0 * Global.reg * Global.reg);
            }

            Global.diff = convergeTest(error);
            return error;
        }

        public double sgd_exactReg()
        {
            double scalar = 1, scalarOld = 1;
            List<double> w = _model.W;
            int fsize = w.Count;
            int xsize = _X.Count;
            double newReg = Global.reg * Math.Sqrt(xsize);
            double oldReg = Global.reg;
            Global.reg = newReg;

            double[] tmpAry = new double[fsize];
            List<double> grad = new List<double>(tmpAry);//store the computed vecGradient

            List<int> ri = randomTool<int>.getShuffledIndexList(xsize);
            double error = 0;
            double r_k = 0;

            for (int t = 0; t < xsize; t++)
            {
                int ii = ri[t];
                dataSeq x = _X[ii];
                baseHashSet<int> fset = new baseHashSet<int>();
                double err = _grad.getGrad_SGD(grad, scalar, _model, x, fset);
                error += err;
                r_k = Global.rate0 * Math.Pow(Global.decayFactor, (double)Global.countWithIter / (double)xsize);
                if (Global.countWithIter % (xsize / 4) == 0)
                    Global.swLog.WriteLine("iter{0}    decay_rate={1}", Global.glbIter, r_k.ToString("e2"));

                //reg
                if (t % Global.scalarResetStep == 0)
                {
                    //reset
                    for (int i = 0; i < fsize; i++)
                        w[i] *= scalar;
                    scalar = scalarOld = 1;
                }
                else
                {
                    scalarOld = scalar;
                    scalar *= 1 - r_k / (Global.reg * Global.reg);
                }

                foreach (int i in fset)
                {
                    double realWeight = w[i] * scalarOld;
                    double grad_i = grad[i] + realWeight / (Global.reg * Global.reg);
                    realWeight = realWeight - r_k * grad_i;
                    w[i] = realWeight / scalar;
                    //reset
                    grad[i] = 0;
                }
                Global.countWithIter++;
            }

            //recover the real weights
            for (int i = 0; i < fsize; i++)
            {
                w[i] *= scalar;
            }

            if (Global.reg != 0.0)
            {
                double sum = listTool.squareSum(w);
                error += sum / (2.0 * Global.reg * Global.reg);
            }

            Global.diff = convergeTest(error);
            Global.reg = oldReg;
            return error;
        }

        public double sgd_multi()
        {
            int fsize = (_modelList[0]).W.Count;
            List<int> sizeList = new List<int>();
            int maxSize = 0;
            for (int i = 0; i < _newXList.Count; i++)
            {
                dataSet Xi = _newXList[i];
                int size = Xi.Count;
                sizeList.Add(size);
                if (maxSize < size)
                    maxSize = size;
            }

            double error = 0;
            double r_k = 0;
            List<double> vecGrad = new List<double>(new double[fsize]);

            List<List<int>> riList = new List<List<int>>();
            for (int i = 0; i < _newXList.Count; i++)
            {
                int size = sizeList[i];
                List<int> ri = randomTool<int>.getShuffledIndexList(size);
                riList.Add(ri);
            }

            for (int t = 0; t < maxSize; t++)
            {
                r_k = Global.rate0 * Math.Pow(Global.decayFactor, (double)Global.countWithIter / (double)maxSize);
                if (Global.countWithIter % (maxSize / 4) == 0)
                    Global.swLog.WriteLine("iter{0}    decay_rate={1}", Global.glbIter, r_k.ToString("e2"));

                List<dataSeq> X = new List<dataSeq>();
                for (int i = 0; i < _newXList.Count; i++)
                {
                    dataSet Xi = _newXList[i];
                    List<int> ri = riList[i];
                    int size = sizeList[i];
                    int idx = ri[t % size];
                    dataSeq x = Xi[idx];
                    X.Add(x);
                }

                baseHashSet<int> fset = new baseHashSet<int>();
                for (int i = 0; i < Global.nTask; i++)
                    for (int j = 0; j < Global.nTask; j++)
                    {
                        model m = _modelList[i];
                        List<double> w = m.W;
                        dataSeq x = X[j];
                        double err = _grad.getGrad_SGD(vecGrad, m, x, fset);
                        if (i == j)
                            error += err;
                        double simi = _simiBiAry[i, j];
                        weightUpdate(w, vecGrad, fset, r_k * simi);
                    }

                Global.countWithIter++;
            }

            //reg
            for (int i = 0; i < Global.nTask; i++)
                error += reg(_modelList[i], fsize, r_k);

            //update the similarity biAry
            if (Global.glbIter == Global.simiUpdateIter)
            {
                if (Global.simiMode == "cov")
                    updateSimi_covariance(_modelList);
                else if (Global.simiMode == "poly")
                    updateSimi_polynomial(_modelList);
                else if (Global.simiMode == "rbf")
                    updateSimi_RBF(_modelList);
                Console.WriteLine("updated simi-matrix!");
            }
            return error;
        }

        //fast multi-task learning via approximation
        public double sgd_multi_fast()
        {
            int fsize = (_modelList[0]).W.Count;
            List<int> sizeList = new List<int>();
            int maxSize = 0;
            for (int i = 0; i < _newXList.Count; i++)
            {
                dataSet Xi = _newXList[i];
                int size = Xi.Count;
                sizeList.Add(size);
                if (maxSize < size)
                    maxSize = size;
            }

            double error = 0;
            double r_k = 0;
            List<double> vecGrad = new List<double>(new double[fsize]);

            List<List<int>> riList = new List<List<int>>();
            for (int i = 0; i < _newXList.Count; i++)
            {
                int size = sizeList[i];
                List<int> ri = randomTool<int>.getShuffledIndexList(size);
                riList.Add(ri);
            }

            for (int t = 0; t < maxSize; t++)
            {
                r_k = Global.rate0 * Math.Pow(Global.decayFactor, (double)Global.countWithIter / (double)maxSize);
                if (Global.countWithIter % (maxSize / 4) == 0)
                    Global.swLog.WriteLine("iter{0}    decay_rate={1}", Global.glbIter, r_k.ToString("e2"));

                List<dataSeq> X = new List<dataSeq>();
                for (int i = 0; i < _newXList.Count; i++)
                {
                    dataSet Xi = _newXList[i];
                    List<int> ri = riList[i];
                    int size = sizeList[i];
                    int idx = ri[t % size];
                    dataSeq x = Xi[idx];
                    X.Add(x);
                }

                baseHashSet<int> fset = new baseHashSet<int>();
                for (int i = 0; i < Global.nTask; i++)
                    for (int j = 0; j < Global.nTask; j++)
                    {
                        if (i == j)
                        {
                            model m = _modelList[i];
                            List<double> w = m.W;
                            dataSeq x = X[j];
                            double err = _grad.getGrad_SGD(vecGrad, m, x, fset);
                            weightUpdate(w, vecGrad, fset, r_k);

                            error += err;
                        }
                        else if (t % Global.sampleFactor == 0)//probabilistic sampling for faster speed
                        {
                            model m = _modelList[i];
                            List<double> w = m.W;
                            dataSeq x = X[j];
                            double err = _grad.getGrad_SGD(vecGrad, m, x, fset);
                            double simi = _simiBiAry[i, j];
                            weightUpdate(w, vecGrad, fset, r_k * simi * Global.sampleFactor);
                        }
                    }

                Global.countWithIter++;
            }

            //reg
            for (int i = 0; i < Global.nTask; i++)
                error += reg(_modelList[i], fsize, r_k);

            //update the similarity biAry
            if (Global.glbIter == Global.simiUpdateIter)
            {
                if (Global.simiMode == "cov")
                    updateSimi_covariance(_modelList);
                else if (Global.simiMode == "poly")
                    updateSimi_polynomial(_modelList);
                else if (Global.simiMode == "rbf")
                    updateSimi_RBF(_modelList);
                Console.WriteLine("updated simi-matrix!");
            }
            return error;
        }

        public void printSimi(double[,] simiAry)
        {
            for (int i = 0; i < Global.nTask; i++)
                for (int j = 0; j < Global.nTask; j++)
                {
                    Global.swSimi.Write(_simiBiAry[i, j].ToString("e3") + ",");
                }
            Global.swSimi.WriteLine();
            Global.swSimi.Flush();
        }

        //compute even when i==j
        public void updateSimi_covariance(List<model> mList)
        {
            model mi;
            model mj;
            for (int i = 0; i < Global.nTask; i++)
                for (int j = 0; j < Global.nTask; j++)
                {
                    if (i == j)
                        continue;
                    mi = mList[i];
                    mj = mList[j];

                    int d = mi.W.Count;
                    List<double> wi = mi.W;
                    List<double> wj = mj.W;

                    //E(x*y)-E(x)*E(y)
                    double Exy = 0, Ex = 0, Ey = 0, Ex2 = 0, Ey2 = 0;
                    for (int k = 0; k < d; k++)
                    {
                        Exy += wi[k] * wj[k];
                        Ex += wi[k];
                        Ey += wj[k];
                        Ex2 += wi[k] * wi[k];
                        Ey2 += wj[k] * wj[k];
                    }
                    Exy /= d;
                    Ex /= d;
                    Ey /= d;
                    Ex2 /= d;
                    Ey2 /= d;
                    double sigma_x = Math.Sqrt(Ex2 - Ex * Ex);
                    double sigma_y = Math.Sqrt(Ey2 - Ey * Ey);

                    double cor = (Exy - Ex * Ey) / (sigma_x * sigma_y);
                    cor /= Global.C;

                    if (cor < 0)
                        cor = 0;
                    _simiBiAry[i, j] = cor;
                }
            printSimi(_simiBiAry);
        }

        public void updateSimi_polynomial(List<model> mList)
        {
            model mi;
            model mj;
            for (int i = 0; i < Global.nTask; i++)
                for (int j = 0; j < Global.nTask; j++)
                {
                    if (i == j)
                        continue;
                    mi = mList[i];
                    mj = mList[j];

                    double kern = 0.0;
                    int d = mi.W.Count;
                    List<double> wi = mi.W;
                    List<double> wj = mj.W;

                    double miNorm = 0, mjNorm = 0, innerProduct = 0;
                    for (int k = 0; k < d; k++)
                    {
                        miNorm += wi[k] * wi[k];
                        mjNorm += wj[k] * wj[k];
                        innerProduct += wi[k] * wj[k];
                    }
                    miNorm = Math.Sqrt(miNorm);
                    mjNorm = Math.Sqrt(mjNorm);
                    kern = innerProduct / (miNorm * mjNorm);

                    kern /= Global.C;
                    if (kern < 0)
                        kern = 0;
                    _simiBiAry[i, j] = kern;
                }
            printSimi(_simiBiAry);
        }

        public void updateSimi_RBF(List<model> mList)
        {
            model mi;
            model mj;
            for (int i = 0; i < Global.nTask; i++)
                for (int j = 0; j < Global.nTask; j++)
                {
                    if (i == j)
                        continue;
                    mi = mList[i];
                    mj = mList[j];

                    double kern = 0.0;
                    int d = mi.W.Count;
                    List<double> wi = mi.W;
                    List<double> wj = mj.W;

                    double miNorm = 0, mjNorm = 0, sum = 0;
                    for (int k = 0; k < d; k++)
                    {
                        miNorm += wi[k] * wi[k];
                        mjNorm += wj[k] * wj[k];
                        double diff = wi[k] - wj[k];
                        sum += diff * diff;
                    }
                    double norm = Math.Sqrt(miNorm * mjNorm);
                    kern = Math.Exp(-sum / norm);

                    kern /= Global.C;
                    if (kern < 0)
                        kern = 0;
                    _simiBiAry[i, j] = kern;
                }
            printSimi(_simiBiAry);
        }

        public void weightUpdate(List<double> w, List<double> grad, baseHashSet<int> idSet, double rs)
        {
            foreach (int i in idSet)
            {
                //minus the gradient to find the minumum point
                w[i] -= rs * grad[i];
                //reset
                grad[i] = 0;
            }
        }

        public double reg(model m, int nFeatures, double r_k)
        {
            double error = 0;
            if (Global.reg != 0.0)
            {
                for (int i = 0; i < nFeatures; i++)
                {
                    double grad_i = m.W[i] / (Global.reg * Global.reg);
                    m.W[i] -= r_k * grad_i;
                }

                if (Global.reg != 0.0)
                {
                    List<double> tmpWeights = m.W;
                    double sum = listTool.squareSum(tmpWeights);
                    error += sum / (2.0 * Global.reg * Global.reg);
                }
            }
            return error;
        }

    }
}