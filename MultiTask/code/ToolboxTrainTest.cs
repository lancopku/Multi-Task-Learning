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
    class toolbox
    {
        protected dataSet _X;
        protected List<dataSet> _XList;//for multi-task
        protected model _model;
        protected List<model> _modelList;//for multi-task
        protected Optimizer _optim;
        protected inference _inf;
        protected featureGenerator _fGene;
        protected gradient _grad;

        public toolbox()
        {
        }

        //for single-task
        public toolbox(dataSet X, bool train=true)
        {
            if (train)//to train
            {
                _XList = null;
                _modelList = null;
                _X = X;
                _fGene = new featureGenerator(X);
                _model = new model(X, _fGene);
                _inf = new inference(this);
                _grad = new gradient(this);
                initOptimizer();
            }
            else//to test
            {
                _XList = null;
                _modelList = null;
                _X = X;
                _model = new model(Global.modelDir + Global.fModel);
                _fGene = new featureGenerator(X);
                _inf = new inference(this);
                _grad = new gradient(this);
            }
        }

        //for multi-task
        public toolbox(dataSet X, List<dataSet> XList, bool train=true)
        {
            if (train)//to train
            {
                _X = X;
                _XList = XList;
                _fGene = new featureGenerator(X);
                _model = null;
                _modelList = new List<model>();
                for (int i = 0; i < Global.nTask; i++)
                {
                    model m = new model(XList[i], _fGene);
                    _modelList.Add(m);
                }
                _inf = new inference(this);
                _grad = new gradient(this);
                initOptimizer();
            }
            else//to test
            {
                _X = X;
                _XList = XList;
                _model = null;
                _modelList = new List<model>();
                for (int i = 0; i < Global.nTask; i++)
                {
                    model m = new model(Global.modelDir + i.ToString() + Global.fModel);
                    _modelList.Add(m);
                }
                _fGene = new featureGenerator(X);
                _inf = new inference(this);
                _grad = new gradient(this);
            }
        }

        public void initOptimizer()
        {
            if (Global.optim.Contains("sgd"))
            {
                _optim = new optimSGD(this);
            }
            else if (Global.optim.Contains("bfgs"))
            {
                List<double> init = new List<double>(_model.W);
                _optim = new optimLBFGS(this, init, Global.mBFGS, 0, Global.ttlIter);
            }
            else throw new Exception("error");
        }

        public double train_single()
        {
            Global.swLog.WriteLine("iter:  " + Global.glbIter.ToString());
            Global.swLog.Flush();
            Console.WriteLine("iter: " + Global.glbIter.ToString());
            //start training
            return _optim.optimize();
        }

        public double train_multi()
        {
            Global.swLog.WriteLine("iter: " + Global.glbIter.ToString());
            Global.swLog.Flush();
            Console.WriteLine("iter: " + Global.glbIter.ToString());
            //start training
            return _optim.optimize_multi();
        }

        public List<double> test_single(dataSet XX, double iter, StreamWriter swOutput)
        {
            List<double> scoreList;
            if (Global.evalMetric == "tok.acc")
                scoreList = decode_tokAcc(XX, _model, iter, swOutput);
            else if (Global.evalMetric == "str.acc")
                scoreList = decode_strAcc(XX, _model, iter, swOutput);
            else if (Global.evalMetric == "f1")
                scoreList = decode_fscore(XX, _model, iter, swOutput);
            else throw new Exception("error");
            return scoreList;
        }

        public List<double> test_multi_merge(List<dataSet> XXList, double iter, List<StreamWriter> swOutputList)
        {
            List<double> scoreList = new List<double>();
            for (int i = 0; i < XXList.Count; i++)
            {
                dataSet X = XXList[i];
                List<double> scoreList_i;
                if (Global.evalMetric == "tok.acc")
                    scoreList_i = decode_tokAcc(X, _model, iter, swOutputList[i]);
                else if (Global.evalMetric == "str.acc")
                    scoreList_i = decode_strAcc(X, _model, iter, swOutputList[i]);
                else if (Global.evalMetric == "f1")
                    scoreList_i = decode_fscore(X, _model, iter, swOutputList[i]);
                else throw new Exception("error");
                scoreList.Add(scoreList_i[0]);
            }
            return scoreList;
        }

        public List<double> test_multi_mtl(List<dataSet> XXList, double iter, List<StreamWriter> swOutputList)
        {
            List<double> scoreList = new List<double>();
            for (int i = 0; i < XXList.Count; i++)
            {
                dataSet X = XXList[i];
                model m = _modelList[i];
                List<double> scoreList_i;
                if (Global.evalMetric == "tok.acc")
                    scoreList_i = decode_tokAcc(X, m, iter, swOutputList[i]);
                else if (Global.evalMetric == "str.acc")
                    scoreList_i = decode_strAcc(X, m, iter, swOutputList[i]);
                else if (Global.evalMetric == "f1")
                    scoreList_i = decode_fscore(X, m, iter, swOutputList[i]);
                else throw new Exception("error");
                scoreList.Add(scoreList_i[0]);
            }
            return scoreList;
        }

        //test2 mode in multi-task test: get the most similar task/model for test 
        public List<double> test2_multi_mtl(List<List<double>> vecList, List<dataSet> XXList, double iter, List<StreamWriter> swOutputList)
        {
            List<double> scoreList = new List<double>();
            for (int i = 0; i < XXList.Count; i++)
            {
                dataSet X = XXList[i];
                List<double> vec = MainClass.getVecFromX(X);
                double cos = -2;
                int idx = -1;
                for (int j = 0; j < vecList.Count; j++)
                {
                    double newCos = mathTool.cos(vecList[j], vec);
                    if (newCos > cos)
                    {
                        idx = j;
                        cos = newCos;
                    }
                }

                model m = _modelList[idx];
                List<double> scoreList_i;
                if (Global.evalMetric == "tokAcc")
                    scoreList_i = decode_tokAcc(X, m, iter, swOutputList[i]);
                else if (Global.evalMetric == "strAcc")
                    scoreList_i = decode_strAcc(X, m, iter, swOutputList[i]);
                else if (Global.evalMetric == "f1")
                    scoreList_i = decode_fscore(X, m, iter, swOutputList[i]);
                else throw new Exception("error");
                scoreList.Add(scoreList_i[0]);
            }
            return scoreList;
        }
        
        //test3 mode in multi-task test: all models vote
        public List<double> test3_multi_mtl(List<List<double>> vecList, List<dataSet> XXList, double iter, List<StreamWriter> swOutputList)
        {
            List<double> scoreList = new List<double>();
            for (int i = 0; i < XXList.Count; i++)
            {
                dataSet X = XXList[i];
                List<double> vec = MainClass.getVecFromX(X);
                model m = new model(_modelList[0], false);
                double ttlCos = 0;
                for (int j = 0; j < vecList.Count; j++)
                {
                    double cos = mathTool.cos(vecList[j], vec);
                    for (int k = 0; k < m.W.Count; k++)
                    {
                        m.W[k] += cos * _modelList[j].W[k];
                        ttlCos += cos;
                    }
                }
                for (int k = 0; k < m.W.Count; k++)
                {
                    m.W[k] /= ttlCos;
                }

                List<double> scoreList_i;
                if (Global.evalMetric == "tok.acc")
                    scoreList_i = decode_tokAcc(X, m, iter, swOutputList[i]);
                else if (Global.evalMetric == "str.acc")
                    scoreList_i = decode_strAcc(X, m, iter, swOutputList[i]);
                else if (Global.evalMetric == "f1")
                    scoreList_i = decode_fscore(X, m, iter, swOutputList[i]);
                else throw new Exception("error");
                scoreList.Add(scoreList_i[0]);
            }
            return scoreList;
        }

        public List<double> decode_tokAcc(dataSet XX, model m, double iter, StreamWriter swOutput)
        {
            int nTag = m.NTag;
            int[] tmpAry = new int[nTag];
            List<int> corrOutput = new List<int>(tmpAry);
            List<int> gold = new List<int>(tmpAry);
            List<int> output = new List<int>(tmpAry);

            foreach (dataSeq x in XX)
            {
                List<int> tags = new List<int>();
                double prob = _inf.decodeViterbi(m, x, tags);

                //output result tags
                if (swOutput != null)
                {
                    for (int i = 0; i < x.Count; i++)
                    {
                        swOutput.Write(tags[i] + ",");
                    }
                    swOutput.WriteLine();
                }

                //count tags for the sample
                for (int i = 0; i < x.Count; i++)
                {
                    gold[x.getTags(i)]++;
                    output[tags[i]]++;

                    if (tags[i] == x.getTags(i))
                        corrOutput[tags[i]]++;
                }
            }
            
            double prec, recall;
            int sumGold = 0, sumOutput = 0, sumCorrOutput = 0;
            for (int i = 0; i < nTag; i++)
            {
                sumCorrOutput += corrOutput[i];
                sumGold += gold[i];
                sumOutput += output[i];
            }
            if (sumGold == 0)
                recall = 0;
            else
                recall = ((double)sumCorrOutput) * 100.0 / (double)sumGold;
            if (sumOutput == 0)
                prec = 0;
            else
                prec = ((double)sumCorrOutput) * 100.0 / (double)sumOutput;
            double fscore;
            if (prec == 0 && recall == 0)
                fscore = 0;
            else
                fscore = 2 * prec * recall / (prec + recall);
            List<double> scoreList = new List<double>();
            scoreList.Add(fscore);
            return scoreList;
        }

        public List<double> decode_strAcc(dataSet XX, model m, double iter, StreamWriter swOutput)
        {
            int nTag = m.NTag;
            double ttl = XX.Count;
            double correct = 0;

            foreach (dataSeq x in XX)
            {
                //compute detected tags
                List<int> tags = new List<int>();
                double prob = _inf.decodeViterbi(m, x, tags);

                //output result tags
                if (swOutput != null)
                {
                    for (int i = 0; i < x.Count; i++)
                    {
                        swOutput.Write(tags[i] + ",");
                    }
                    swOutput.WriteLine();
                }

                List<int> goldTags = x.getTags();
                bool ck = true;
                for (int i = 0; i < x.Count; i++)
                {
                    if (goldTags[i] != tags[i])
                    {
                        ck = false;
                        break;
                    }
                }
                if (ck)
                    correct++;
            }
            double acc = correct / ttl;
            List<double> scoreList = new List<double>();
            scoreList.Add(acc);
            return scoreList;
        }

        public List<double> decode_fscore(dataSet XX, model m, double iter, StreamWriter swOutput)
        {
            int nTag = m.NTag;
            double ttl = XX.Count;

            List<string> goldTagList = new List<string>();
            List<string> resTagList = new List<string>();
            foreach (dataSeq x in XX)
            {
                //compute detected tags
                List<int> tags = new List<int>();
                double prob = _inf.decodeViterbi(m, x, tags);

                string res = "";
                foreach (int im in tags)
                    res += im.ToString() + ",";
                resTagList.Add(res);

                //output result tags
                if (swOutput != null)
                {
                    for (int i = 0; i < x.Count; i++)
                    {
                        swOutput.Write(tags[i] + ",");
                    }
                    swOutput.WriteLine();
                }

                List<int> goldTags = x.getTags();
                string gold = "";
                foreach (int im in goldTags)
                    gold += im.ToString() + ",";
                goldTagList.Add(gold);
            }
            List<double> infoList = new List<double>();
            List<double> scoreList = fscore.getFscore(goldTagList, resTagList, infoList);
            return scoreList;
        }

        public model Model
        {
            get { return _model; }
        }

        public List<model> ModelList
        {
            get { return _modelList; }
        }

        public dataSet X
        {
            get { return _X; }
        }

        public List<dataSet> XList
        {
            get { return _XList; }
        }

        public inference Inf
        {
            get { return _inf; }
        }

        public gradient Grad
        {
            get { return _grad; }
        }

        public featureGenerator FGene
        {
            get { return _fGene; }

        }

        public Optimizer Optim
        {
            get { return _optim; }
        }
    }
}