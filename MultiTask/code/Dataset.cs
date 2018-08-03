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
using System.IO.Compression;

namespace Program
{
    class dsList:List<dataSet>
    {
        protected int _nTag;
        protected int _nFeature;

        public dsList(string fileFeature, string fileTags)
        {
            StreamReader srfileFeature = new StreamReader(fileFeature);
            StreamReader srfileTags = new StreamReader(fileTags);

            string txt = srfileFeature.ReadToEnd();
            txt = txt.Replace("\r", "");
            string[] fAry = txt.Split(Global.triLnEndAry, StringSplitOptions.RemoveEmptyEntries);
            
            txt = srfileTags.ReadToEnd();
            txt = txt.Replace("\r", "");
            string[] tAry = txt.Split(Global.triLnEndAry, StringSplitOptions.RemoveEmptyEntries);

            if (fAry.Length != tAry.Length)
                throw new Exception("error");

            _nFeature = int.Parse(fAry[0]);
            _nTag = int.Parse(tAry[0]);
            
            for (int i = 1; i < fAry.Length; i++)
            {
                string fBlock = fAry[i];
                string tBlock = tAry[i];
                dataSet ds = new dataSet();
                string[] fbAry = fBlock.Split(Global.biLnEndAry, StringSplitOptions.RemoveEmptyEntries);
                string[] lbAry = tBlock.Split(Global.biLnEndAry, StringSplitOptions.RemoveEmptyEntries);

                for (int k = 0; k < fbAry.Length; k++)
                {
                    string fm = fbAry[k];
                    string lm = lbAry[k];
                    dataSeq seq = new dataSeq();
                    seq.read(fm, lm);
                    ds.Add(seq);
                }
                Add(ds);
            }
            srfileFeature.Close();
            srfileTags.Close();
        }
    }

    class dataSet : List<dataSeq>
    {
        protected int _nTag;
        protected int _nFeatureTemp;

        public dataSet()
        {
        }

        public dataSet(int nTag, int nFeatureTemp)
        {
            _nTag = nTag;
            _nFeatureTemp = nFeatureTemp;
        }

        public dataSet(string fileFeature, string fileTags)
        {
            load(fileFeature, fileTags);
        }

        virtual public int[,] EdgeFeature()
        {
            throw new Exception("error");
        }

        virtual public void load(string fileFeature, string fileTag)
        {
            StreamReader srfileFeature = new StreamReader(fileFeature, Encoding.GetEncoding("utf-8"));
            StreamReader srfileTag = new StreamReader(fileTag, Encoding.GetEncoding("utf-8"));

            string txt = srfileFeature.ReadToEnd();
            txt = txt.Replace("\r", "");
            string[] fAry = txt.Split(Global.biLnEndAry, StringSplitOptions.RemoveEmptyEntries);

            txt = srfileTag.ReadToEnd();
            txt = txt.Replace("\r", "");
            string[] tAry = txt.Split(Global.biLnEndAry, StringSplitOptions.RemoveEmptyEntries);

            if (fAry.Length != tAry.Length)
                throw new Exception("error");

            _nFeatureTemp = int.Parse(fAry[0]);
            _nTag = int.Parse(tAry[0]);
            for (int i = 1; i < fAry.Length; i++)
            {
                string features = fAry[i];
                string tags = tAry[i];
                dataSeq seq = new dataSeq();
                seq.read(features, tags);
                Add(seq);
            }
            srfileFeature.Close();
            srfileTag.Close();
        }

        public int NTag
        {
            get { return _nTag; }
            set { _nTag = value; }
        }

        public int NFeatureTemp
        {
            get { return _nFeatureTemp; }
            set { _nFeatureTemp = value; }
        }

        public void setDataInfo(dataSet X)
        {
            _nTag = X.NTag;
            _nFeatureTemp = X.NFeatureTemp;
        }

    }

    class dataSeq
    {
        protected List<List<featureTemp>> featureTemps = new List<List<featureTemp>>();
        protected List<string> yGold = new List<string>();
        protected dMatrix goldStatesPerNode = new dMatrix();

        public dataSeq()
        {
        }

        public dataSeq(List<List<featureTemp>> feat, List<string> y)
        {
            featureTemps = new List<List<featureTemp>>(feat);
            for (int i = 0; i < feat.Count; i++)
                featureTemps[i] = new List<featureTemp>(feat[i]);
            yGold = new List<string>(y);
        }

        public dataSeq(List<List<featureTemp>> feat, List<int> y)
        {
            featureTemps = new List<List<featureTemp>>(feat);
            for (int i = 0; i < feat.Count; i++)
                featureTemps[i] = new List<featureTemp>(feat[i]);
            yGold = new List<string>(new string[y.Count]);
            for (int i = 0; i < y.Count; i++)
                yGold[i] = y[i].ToString();
        }

        virtual public List<List<int>> getNodeFeature(int n)
        {
            throw new Exception("error");
        }

        virtual public void read(string a, int nState, string b)
        {
            throw new Exception("error");
        }

        public void read(string a, string b)
        {
            //features
            string[] lineAry = a.Split(Global.lineEndAry, StringSplitOptions.RemoveEmptyEntries);
            foreach (string im in lineAry)
            {
                List<featureTemp> nodeList = new List<featureTemp>();
                string[] imAry = im.Split(Global.commaAry, StringSplitOptions.RemoveEmptyEntries);
                foreach (string imm in imAry)
                {
                    if (imm.Contains("/"))
                    {
                        string[] biAry = imm.Split(Global.slashAry, StringSplitOptions.RemoveEmptyEntries);
                        featureTemp ft = new featureTemp(int.Parse(biAry[0]), double.Parse(biAry[1]));
                        nodeList.Add(ft);
                    }
                    else
                    {
                        featureTemp ft = new featureTemp(int.Parse(imm), 1);
                        nodeList.Add(ft);
                    }
                }
                featureTemps.Add(nodeList);
            }

            //yGold
            lineAry = b.Split(Global.commaAry, StringSplitOptions.RemoveEmptyEntries);
            foreach (string im in lineAry)
            {
                yGold.Add(im);
            }
        }

        virtual public int Count
        {
            get { return featureTemps.Count; }
        }

        public List<List<featureTemp>> getFeatureTemp()
        {
            return featureTemps;
        }

        public List<featureTemp> getFeatureTemp(int node)
        {
            return featureTemps[node];
        }

        public int getTags(int node)
        {
            if (!yGold[node].Contains("/"))
                return int.Parse(yGold[node]);
            else
                throw new Exception("error");
        }

        public List<int> getTags()
        {
            List<int> list = new List<int>(new int[yGold.Count]);
            for (int i = 0; i < yGold.Count; i++)
                list[i] = int.Parse(yGold[i]);
            return list;
        }

        public void setTags(List<int> list)
        {
            if (list.Count != yGold.Count)
                throw new Exception("error");
            for (int i = 0; i < list.Count; i++)
                yGold[i] = list[i].ToString();
        }

        public dMatrix GoldStatesPerNode
        {
            get{return goldStatesPerNode;}
            set { goldStatesPerNode = value; }
        }
            
    }














}