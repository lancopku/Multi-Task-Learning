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
    class listTool
    {
        public static List<double> getDoubleList(string a)
        {
            string[] tokens = a.Split(Global.commaAry, StringSplitOptions.RemoveEmptyEntries);
            List<double> dList = new List<double>();
            foreach (string token in tokens)
            {
                dList.Add(double.Parse(token));
            }
            return dList;
        }

        public static double squareSum(List<double> list)
        {
            double sum = 0;
            foreach (double im in list)
                sum += im * im;
            return sum;
        }

        public static void listSet(ref List<double> a, List<double> b)
        {
            if (a.Count != b.Count)
                throw new Exception("error");
            for (int i = 0; i < a.Count; i++)
                a[i] = b[i];
        }

        public static void listSet_add(ref List<double> a, List<double> b)
        {
            a.Clear();
            foreach (double im in b)
                a.Add(im);
        }

        public static void listSet(ref List<double> a, double v)
        {
            for (int i = 0; i < a.Count; i++)
                a[i] = v;
        }

        public static void listAdd(ref List<double> a, List<double> b)
        {
            if (a.Count != b.Count)
                throw new Exception("error");
            for (int i = 0; i < a.Count; i++)
                a[i] += b[i];
        }

        public static void listAdd(ref List<double> a, double v)
        {
            for (int i = 0; i < a.Count; i++)
                a[i] += v;
        }

        public static void listMultiply(ref List<double> a, double v)
        {
            for (int i = 0; i < a.Count; i++)
                a[i] *= v;
        }

        public static void listExp(ref List<double> a)
        {
            for (int i = 0; i < a.Count; i++)
                a[i] = Math.Exp(a[i]);
        }

        public static void listSwap(ref List<double> a, ref List<double> b)
        {
            if (a.Count != b.Count)
                throw new Exception("error");
            for (int i = 0; i < a.Count; i++)
            {
                double tmp = a[i];
                a[i] = b[i];
                b[i] = tmp;
            }
        }

        public static void showSample(List<double> a)
        {
            int L = a.Count;
            int step = L / 10;
            for (int i = 0; i < L; i += step)
                Console.WriteLine("#" + i.ToString() + " " + a[i].ToString("e2"));
            Console.WriteLine();
        }

        public static void showSample(StreamWriter sw, List<double> a)
        {
            int L = a.Count;
            int step = L / 10;
            for (int i = 0; i < L; i += step)
                sw.WriteLine("#" + i.ToString() + " " + a[i].ToString("e2"));
            sw.WriteLine();
        }

        public static void showSample(List<int> a)
        {
            int L = a.Count;
            int step = L / 10;
            if (step == 0)
                step = 1;
            for (int i = 0; i < L; i += step)
                Console.WriteLine("#" + i.ToString() + " " + a[i].ToString("e2"));
            Console.WriteLine();
        }
    }

    class fileTool
    {
        public static void removeFile(string folder)
        {
            string[] files = Directory.GetFiles(folder);
            foreach (string file in files)
            {
                File.Delete(file);
            }
        }
    }

    class ListSortFunc
    {
        public static int compareKVpair(string a, string b)
        {
            string[] aAry = a.Split(Global.tabAry, StringSplitOptions.None);
            string[] bAry = b.Split(Global.tabAry, StringSplitOptions.None);
            double aProb = double.Parse(aAry[aAry.Length - 1]);
            double bProb = double.Parse(bAry[bAry.Length - 1]);
            if (aProb < bProb)
                return 1;
            else if (aProb > bProb)
                return -1;
            else return 0;

        }
    }

    class ordinalComparerTool : IComparer<string>
    {
        public int Compare(string x, string y)
        {
            return (string.CompareOrdinal(x, y));
        }
    }

    class stringTool
    {
        int _index;

        public stringTool()
        {
            _index = 0;
        }

        public static string charAry2str(char[] cAry)
        {
            string[] sAry = new string[cAry.Length];
            for (int i = 0; i < cAry.Length; i++)
                sAry[i] = cAry[i].ToString();
            return string.Join("", sAry);
        }

        public static void readToEndSB(StreamReader sr, ref StringBuilder output)
        {
            while (!(sr.EndOfStream))
            {
                string line = sr.ReadLine();
                output.AppendLine(line);
            }
        }

        public static int stringCount(string a, string mark, ref List<int> posList)
        {
            int count = 0;
            int i = 0;
            while (true)
            {
                i = a.IndexOf(mark, i);
                if (i != -1)
                {
                    count++;
                    posList.Add(i);
                }
                else
                    break;
                i++;
            }
            return count;
        }

        public bool nextSubstring(string a, string mark, ref string output)
        {
            if (_index == -1)
                return false;
            int nextIndex = a.IndexOf(mark, _index);
            if (nextIndex != -1)
            {
                output = a.Substring(_index, nextIndex - _index);
                _index = nextIndex + mark.Length;
            }
            else
            {
                output = a.Substring(_index);
                _index = -1;
            }
            return true;
        }
    }

    //T can be all kinds of numbers: int, double, float, etc
    class randomTool<T>
    {
        public static List<T> randomShuffle(List<T> list)
        {
            Random rand = new Random();
            SortedDictionary<int, T> sortMap = new SortedDictionary<int, T>();
            foreach (T im in list)
            {
                int rdInt = rand.Next();
                while (sortMap.ContainsKey(rdInt))
                    rdInt = rand.Next();
                sortMap.Add(rdInt, im);
            }
            List<T> newList = new List<T>();
            foreach (KeyValuePair<int, T> im in sortMap)
            {
                newList.Add(im.Value);

            }
            return newList;
        }

        public static List<int> getShuffledIndexList(int n)
        {
            int[] dAry = new int[n];
            List<int> ri = new List<int>(dAry);
            for (int j = 0; j < n; j++)
                ri[j] = j;

            ri = randomTool<int>.randomShuffle(ri);
            return ri;
        }

    }

    class randomDoubleTool
    {
        //random double between -1, 1
        public static List<double> getRandomList(int n)
        {
            List<double> list = new List<double>();
            Random rand = new Random();
            for (int i = 0; i < n; i++)
            {
                //a double between 0 and 1
                double val = rand.NextDouble();
                int sign = rand.Next(-10000, 10000);
                if (sign < 0)
                    val *= -1;
                list.Add(val);
            }
            return list;
        }
    }

}
