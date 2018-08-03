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
    class Optimizer
    {
        protected model _model;
        protected List<model> _modelList;
        protected dataSet _X;
        protected List<dataSet> _XList;
        protected inference _inf;
        protected featureGenerator _fGene;
        protected gradient _grad;
       
        //for convergence test
        protected Queue<double> _preVals = new Queue<double>();

        virtual public double optimize()
        {
            throw new Exception("error");
        }

        virtual public double optimize_multi()
        {
            throw new Exception("error");
        }

        public double convergeTest(double err)
        {
            double val = double.MaxValue;
            if (_preVals.Count > 1)
            {
                double prevVal = _preVals.Peek();
                if (_preVals.Count == 10)
                {
                    double trash = _preVals.Dequeue();
                }
                double averageImprovement = (prevVal - err) / _preVals.Count;
                double relAvgImpr = averageImprovement / Math.Abs(err);
                val = relAvgImpr;
            }
            _preVals.Enqueue(err);
            return val;
        }
    }
}