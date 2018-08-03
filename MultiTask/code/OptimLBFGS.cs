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

    class PointValueDeriv
    {
        public double _a, _v, _d;

        public PointValueDeriv()
        {
        }

        public PointValueDeriv(double a, double value, double deriv)
        {
            _a = a;
            _v = value;
            _d = deriv;
        }
    }

    class optimLBFGS : Optimizer
    {
        class DblVecPtrDeque : Queue<List<double>>
        {
        }

        double _maxIter;
        List<double> _w = new List<double>(), _gradList = new List<double>(), _newW = new List<double>(), _newGradList = new List<double>(), _dir = new List<double>();
        List<double> _steepestDescDir;
        DblVecPtrDeque _sList = new DblVecPtrDeque(), yList = new DblVecPtrDeque();
        Queue<double> _roList = new Queue<double>();
        List<double> _alphas = new List<double>();
        double _value;
        int _iter, _memo;
        int _dim;
        double _l1weight;

        public optimLBFGS(toolbox tb, List<double> wInit, int memo, double l1weight, double maxIter)
        {
            _model = tb.Model;
            _modelList = tb.ModelList;
            _X = tb.X;
            _XList = tb.XList;
            _inf = tb.Inf;
            _fGene = tb.FGene;
            _grad = tb.Grad;

            double[] tmpAry = new double[wInit.Count];
            _w = new List<double>(wInit);
            _gradList = new List<double>(tmpAry);
            _newW = new List<double>(wInit);
            _newGradList = new List<double>(tmpAry);
            _dir = new List<double>(tmpAry);
            _steepestDescDir = new List<double>(_newGradList);
            double[] tmpAry2 = new double[memo];
            _alphas = new List<double>(tmpAry2);
            _iter = 0;
            _memo = memo;
            _dim = wInit.Count;
            _l1weight = l1weight;
            _maxIter = maxIter;

            if (memo <= 0)
                throw new Exception("m must be an integer greater than zero.");
            _value = evalL1();
            listTool.listSet(ref _gradList, _newGradList);
        }

        override public double optimize()
        {
            int size = _model.W.Count;
            double[] tmpAry = new double[size];
            List<double> ans = new List<double>(tmpAry);
            Global.swLog.WriteLine("L2 reg: {0}", Global.reg);

            double error = minimize(ans);

            int nonZero = 0;
            for (int i = 0; i < ans.Count; i++)
            {
                if (ans[i] != 0) nonZero++;
            }
            Global.swLog.WriteLine("Finished with optimization.  {0}/{1} non-zero weights.", nonZero, size);

            //save the optimal weights
            _model.W = ans;
            return error;
        }

        override public double optimize_multi()
        {
            throw new Exception("error");
            //return -1;
        }

        public double minimize(List<double> w)
        {
            Global.swLog.WriteLine("L1 reg: {0}", _l1weight);
            Global.swLog.WriteLine("L-BFGS memory parameter (m): {0}", _memo);
            Global.swLog.WriteLine("converge tolerance: {0}", Global.convergeTol.ToString("e2"));
            Global.swLog.WriteLine("#iter   objective   converge-test");

            while (true)
            {
                Stopwatch timer = new Stopwatch();
                timer.Start();

                updateDir();
                if (Global.wolfe)
                    wolfeLineSearch();
                else
                    backTrackLineSearch();
                Global.diff = convergeTest(_value);

                timer.Stop();
                double time = timer.ElapsedMilliseconds / 1000.0;
                Global.swLog.WriteLine("training used time (second): " + time.ToString());

                if (Global.bfgsTestMode == "mt.merge")
                {
                    List<double> scoreList = Global.bfgsTb.test_multi_merge(Global.bfgsXXList, _iter, Global.swOutputList);
                    for (int i = 0; i < Global.nTask; i++)
                        Global.scoreTaskList_multi[i].Add(scoreList[i]);
                    Global.timeList_multi[_iter] += time;
                    Global.errorList_multi[_iter] += _value;
                }
                else if (Global.bfgsTestMode == "mt.single")
                {
                    List<double> resList = Global.bfgsTb.test_single(Global.bfgsXXList[Global.bfgsTaskID], _iter, Global.swOutputList[Global.bfgsTaskID]);
                    Global.scoreTaskList_multi[Global.bfgsTaskID].Add(resList[0]);
                    Global.timeList_multi[_iter] += time;
                    Global.errorList_multi[_iter] += _value;
                }

                Global.swLog.WriteLine("iter" + _iter.ToString() + "   " + _value.ToString("e2") + "   " + Global.diff.ToString("e2"));
                Global.swLog.Flush();
                _iter++;
                if (_iter >= _maxIter || Global.diff < Global.convergeTol)
                    break;

                shift();//should be the last one, should behind test() & break-check
            }
            Global.swLog.WriteLine();
            listTool.listSet(ref w, _newW);
            return _value;
        }

        //compute the loss-value and the gradient given the current weight
        public double getLossGradient(List<double> input, List<double> gradient)
        {
            int size = gradient.Count;
            _model.W = input;

            //important: to clean the values of gradient
            listTool.listSet(ref gradient, 0.0);
            double error = _grad.getGrad_BFGS(gradient, _model, _X);
            return error;
        }

        //to deal with convex function
        public void backTrackLineSearch()
        {
            double origDirDeriv = getDirDeriv();
            if (origDirDeriv >= 0)
            {
                Global.swLog.WriteLine("L-BFGS chose a non-descent direction: check your gradient!");
                throw new Exception("error");
            }

            double alpha = 1.0;
            double backoff = 0.5;
            if (_iter == 0)
            {
                double normDir = Math.Sqrt(dotProduct(_dir, _dir));
                alpha = (1 / normDir);
                backoff = 0.1;
            }

            const double c1 = 1e-4;
            double oldValue = _value;

            while (true)
            {
                getNextPoint(alpha);
                _value = evalL1();
                if (_value <= oldValue + c1 * origDirDeriv * alpha) break;
                Global.swLog.Write(".");
                Console.Write(".");
                alpha *= backoff;
            }
        }

        //to deal with non-convex objective function
        public void wolfeLineSearch()
        {
            double dirDeriv = getDirDeriv();
            double normDir = Math.Sqrt(dotProduct(_dir, _dir));
            if (dirDeriv > 0)
                Global.swLog.WriteLine("L-BFGS edition chose a non-descent direction: check your gradient!");

            double c1 = 1e-4 * dirDeriv;
            double c2 = 0.9 * dirDeriv;
            double a = (_iter == 0 ? (1 / normDir) : 1.0);
            PointValueDeriv last = new PointValueDeriv(0, _value, dirDeriv);
            PointValueDeriv aLo = new PointValueDeriv(), aHi = new PointValueDeriv();
            bool done = false;

            double unitRoundoff = 1e-6;
            if (a * normDir < unitRoundoff)
                Global.swLog.WriteLine("Obtained step size near limits of numerical stability.");

            Global.swLog.WriteLine(" ");

            double newValue = 0;
            while (true)
            {
                getNextPoint2(a);
                newValue = getLossGradient(_newW, _newGradList);
                double oldValue = _value;
                _value = newValue;
                dirDeriv = getNewDirDeriv();
                PointValueDeriv curr = new PointValueDeriv(a, newValue, dirDeriv);
                if ((curr._v > oldValue + c1 * a) || (last._a > 0 && curr._v >= last._v))
                {
                    aLo = last;
                    aHi = curr;
                    break;
                }
                else if (Math.Abs(curr._d) <= -c2)
                {
                    done = true;
                    break;
                }
                else if (curr._d >= 0)
                {
                    aLo = curr;
                    aHi = last;
                    break;
                }
                last = curr;
                a *= 2;
                Global.swLog.Write("+");
            }
            double minChange = 0.01;
            while (!done)
            {
                Global.swLog.Write("-");
                PointValueDeriv left = aLo._a < aHi._a ? aLo : aHi;
                PointValueDeriv right = aLo._a < aHi._a ? aHi : aLo;
                if (left._d > 0 && right._d < 0)
                {
                    a = aLo._v < aHi._v ? aLo._a : aHi._a;
                }
                else
                {
                    a = cubicInterp(aLo, aHi);
                }

                double ub = (minChange * left._a + (1 - minChange) * right._a);
                if (a > ub) a = ub;
                double lb = (minChange * right._a + (1 - minChange) * left._a);
                if (a < lb) a = lb;

                getNextPoint2(a);
                newValue = getLossGradient(_newW, _newGradList);
                double oldValue = _value;
                _value = newValue;
                dirDeriv = getNewDirDeriv();
                PointValueDeriv curr = new PointValueDeriv(a, newValue, dirDeriv);
                if ((curr._v > oldValue + c1 * a) || (curr._v >= aLo._v))
                {
                    aHi = curr;
                }
                else if (Math.Abs(curr._d) <= -c2)
                {
                    done = true;
                }
                else
                {
                    if (curr._d * (aHi._a - aLo._a) >= 0) aHi = aLo;
                    aLo = curr;
                }

                if (aLo._a == aHi._a)
                {
                    Global.swLog.WriteLine("Step size interval numerically zero.");
                }
            }
        }

        public void updateDir()
        {
            makeSteepestDescDir();
            mapDirByInverseHessian();
            fixDirSigns();

            if (Global.debugMode)
                testDirDeriv();
        }

        public void testDirDeriv()
        {
            double dirNorm = Math.Sqrt(dotProduct(_dir, _dir));
            double eps = 1.05e-8 / dirNorm;
            getNextPoint(eps);
            double val2 = evalL1();
            double numDeriv = (val2 - _value) / eps;
            double deriv = getDirDeriv();
            Global.swLog.WriteLine("Grad check: " + numDeriv.ToString("e3") + " vs. " + deriv.ToString("e3") + "  ");
            Console.WriteLine("Grad check: " + numDeriv.ToString("e3") + " vs. " + deriv.ToString("e3") + "  ");
        }

        public void getNextPoint2(double alpha)
        {
            addMultInto(_newW, _w, _dir, alpha);
        }

        public double getDirDeriv()
        {
            return dotProduct(_dir, _gradList);
        }

        public double getNewDirDeriv()
        {
            return dotProduct(_dir, _newGradList);
        }

        public double cubicInterp(PointValueDeriv p0, PointValueDeriv p1)
        {
            double t1 = p0._d + p1._d - 3 * (p0._v - p1._v) / (p0._a - p1._a);
            double t2 = mathSign(p1._a - p0._a) * Math.Sqrt(t1 * t1 - p0._d * p1._d);
            double num = p1._d + t2 - t1;
            double denom = p1._d - p0._d + 2 * t2;
            return p1._a - (p1._a - p0._a) * num / denom;
        }

        public int mathSign(double a)
        {
            if (a > 0)
                return 1;
            else if (a < 0)
                return -1;
            else
                return 0;
        }

        public double dotProduct(List<double> a, List<double> b)
        {
            double result = 0;
            for (int i = 0; i < a.Count; i++)
            {
                result += a[i] * b[i];
            }
            return result;
        }

        public void addMult(List<double> a, List<double> b, double c)
        {
            for (int i = 0; i < a.Count; i++)
            {
                a[i] += b[i] * c;
            }
        }

        public void add(List<double> a, List<double> b)
        {
            for (int i = 0; i < a.Count; i++)
            {
                a[i] += b[i];
            }
        }

        public void addMultInto(List<double> a, List<double> b, List<double> c, double d)
        {
            for (int i = 0; i < a.Count; i++)
            {
                a[i] = b[i] + c[i] * d;
            }
        }

        public void scale(List<double> a, double b)
        {
            for (int i = 0; i < a.Count; i++)
            {
                a[i] *= b;
            }
        }

        public void scaleInto(List<double> a, List<double> b, double c)
        {
            for (int i = 0; i < a.Count; i++)
            {
                a[i] = b[i] * c;
            }
        }

        public void mapDirByInverseHessian()
        {
            int count = _sList.Count;
            List<double>[] sListAry = _sList.ToArray();
            double[] roListAry = _roList.ToArray();
            List<double>[] yListAry = yList.ToArray();

            if (count != 0)
            {
                for (int i = count - 1; i >= 0; i--)
                {
                    _alphas[i] = -dotProduct(sListAry[i], _dir) / roListAry[i];
                    addMult(_dir, yListAry[i], _alphas[i]);
                }

                List<double> lastY = yListAry[count - 1];
                double yDotY = dotProduct(lastY, lastY);
                double scalar = roListAry[count - 1] / yDotY;
                scale(_dir, scalar);

                for (int i = 0; i < count; i++)
                {
                    double beta = dotProduct(yListAry[i], _dir) / roListAry[i];
                    addMult(_dir, sListAry[i], -_alphas[i] - beta);
                }
            }
        }

        public void makeSteepestDescDir()
        {
            if (_l1weight == 0)
            {
                scaleInto(_dir, _gradList, -1);
            }
            else
            {

                for (int i = 0; i < _dim; i++)
                {
                    if (_w[i] < 0)
                    {
                        _dir[i] = -_gradList[i] + _l1weight;
                    }
                    else if (_w[i] > 0)
                    {
                        _dir[i] = -_gradList[i] - _l1weight;
                    }
                    else
                    {
                        if (_gradList[i] < -_l1weight)
                        {
                            _dir[i] = -_gradList[i] - _l1weight;
                        }
                        else if (_gradList[i] > _l1weight)
                        {
                            _dir[i] = -_gradList[i] + _l1weight;
                        }
                        else
                        {
                            _dir[i] = 0;
                        }
                    }
                }
            }

            _steepestDescDir = _dir;
        }

        public void fixDirSigns()
        {
            if (_l1weight > 0)
            {
                for (int i = 0; i < _dim; i++)
                {
                    if (_dir[i] * _steepestDescDir[i] <= 0)
                    {
                        _dir[i] = 0;
                    }
                }
            }
        }

        public double getDirDeriv2()
        {
            if (_l1weight == 0)
            {
                return dotProduct(_dir, _gradList);
            }
            else
            {
                double val = 0.0;
                for (int i = 0; i < _dim; i++)
                {
                    if (_dir[i] != 0)
                    {
                        if (_w[i] < 0)
                        {
                            val += _dir[i] * (_gradList[i] - _l1weight);
                        }
                        else if (_w[i] > 0)
                        {
                            val += _dir[i] * (_gradList[i] + _l1weight);
                        }
                        else if (_dir[i] < 0)
                        {
                            val += _dir[i] * (_gradList[i] - _l1weight);
                        }
                        else if (_dir[i] > 0)
                        {
                            val += _dir[i] * (_gradList[i] + _l1weight);
                        }
                    }
                }
                return val;
            }
        }

        public void getNextPoint(double alpha)
        {
            addMultInto(_newW, _w, _dir, alpha);
            if (_l1weight > 0)
            {
                for (int i = 0; i < _dim; i++)
                {
                    if (_w[i] * _newW[i] < 0.0)
                    {
                        _newW[i] = 0.0;
                    }
                }
            }
        }

        public double evalL1()
        {
            double val = getLossGradient(_newW, _newGradList);
            if (_l1weight > 0)
            {
                for (int i = 0; i < _dim; i++)
                {
                    val += Math.Abs(_newW[i]) * _l1weight;
                }
            }
            return val;
        }

        public void shift()
        {
            List<double> nextS = null, nextY = null;

            int listSize = _sList.Count;

            if (listSize < _memo)
            {
                try
                {
                    double[] tmpAry = new double[_dim];
                    nextS = new List<double>(tmpAry);
                    nextY = new List<double>(tmpAry);
                }
                catch (Exception)
                {
                    _memo = listSize;
                    if (nextS != null)
                    {
                        nextS = null;
                    }
                }
            }

            if (nextS == null)
            {
                nextS = _sList.Peek();
                List<double> trash = _sList.Dequeue();
                nextY = yList.Peek();
                trash = yList.Dequeue();
                double tras = _roList.Dequeue();
            }

            addMultInto(nextS, _newW, _w, -1);
            addMultInto(nextY, _newGradList, _gradList, -1);
            double ro = dotProduct(nextS, nextY);

            _sList.Enqueue(nextS);
            yList.Enqueue(nextY);
            _roList.Enqueue(ro);

            listTool.listSwap(ref _w, ref _newW);
            listTool.listSwap(ref _gradList, ref _newGradList);
        }
    }
}