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
    class featureTemp
    {
        public readonly int id;
        public readonly double val;

        public featureTemp(int a, double b)
        {
            id = a;
            val = b;
        }
    }

    class featureGenerator
    {
        protected int _nFeatureTemp;
        protected int _nCompleteFeature;
        protected int _backoffEdge;
        protected int _nTag;

        public featureGenerator()
        {
        }

        //for train & test
        public featureGenerator(dataSet X)
        {
            _nFeatureTemp = X.NFeatureTemp;
            _nTag = X.NTag;
            Global.swLog.WriteLine("feature templates: {0}", _nFeatureTemp);

            int nNodeFeature = _nFeatureTemp * _nTag;
            int nEdgeFeature = _nTag * _nTag;
            _backoffEdge = nNodeFeature;
            _nCompleteFeature = nNodeFeature + nEdgeFeature;
            Global.swLog.WriteLine("complete features: {0}", _nCompleteFeature);
        }

        public List<featureTemp> getFeatureTemp(dataSeq x, int node)
        {
            return x.getFeatureTemp(node);
        }

        public int getNodeFeatID(int id, int s)
        {
            return id * _nTag + s;
        }

        virtual public int getEdgeFeatID(int sPre, int s)
        {
            return _backoffEdge + s * _nTag + sPre;
        }

        virtual public int getEdgeFeatID(int id, int sPre, int s)
        {
            throw new Exception("error");
        }

        virtual public void getFeatures(dataSeq x, int node, ref List<List<int>> nodeFeature, ref int[,] edgeFeature)
        {
            throw new Exception("error");
        }

        public int BackoffEdge { get { return _backoffEdge; } }

        public int NCompleteFeature { get { return _nCompleteFeature; } }

    }
}