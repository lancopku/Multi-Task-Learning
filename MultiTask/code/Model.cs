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
    class model
    {
        int _nTag;
        List<double> _w;

        //for test
        public model(string file)
        {
            if (File.Exists(file))
                load(file);
            else throw new Exception("error");
        }

        //for train
        public model(dataSet X, featureGenerator fGen)
        {
            _nTag = X.NTag;
            //default value is 0
            if (Global.random == 0)
            {
                double[] dAry = new double[fGen.NCompleteFeature];
                List<double> w = new List<double>(dAry);
                W = w;
            }
            else if (Global.random == 1)
            {
                List<double> randList = randomDoubleTool.getRandomList(fGen.NCompleteFeature);
                W = randList;
            }
            else throw new Exception("error");
        }

        public model(model m, bool wCopy)
        {
            _nTag = m.NTag;
            if (wCopy)
                _w = new List<double>(m.W);
            else _w = new List<double>(new double[m.W.Count]);
        }

        public void load(string file)
        {
            StreamReader sr = new StreamReader(file);
            string txt = sr.ReadToEnd();
            txt = txt.Replace("\r", "");
            string[] ary = txt.Split(Global.lineEndAry, StringSplitOptions.RemoveEmptyEntries);
            _nTag = int.Parse(ary[0]);
            int wsize = int.Parse(ary[1]);
            _w = new List<double>();
            for (int i = 2; i < ary.Length; i++)
            {
                _w.Add(double.Parse(ary[i]));
            }
            if (_w.Count != wsize)
                throw new Exception("error");

            sr.Close();
        }

        public void save(string file)
        {
            StreamWriter sw = new StreamWriter(file);

            sw.WriteLine(_nTag);
            sw.WriteLine(_w.Count);
            foreach (double im in _w)
            {
                sw.WriteLine(im.ToString("f4"));
            }
            sw.Close();
        }

        public List<double> W
        {
            get { return _w; }
            set 
            {
                if (_w == null)
                {
                    double[] ary = new double[value.Count];
                    _w = new List<double>(ary);
                }
                listTool.listSet(ref _w, value);
            }                 
        }

        public int NTag
        {
            get { return _nTag; }
            set
            {
                _nTag = value;
            }
        }

    }
}
