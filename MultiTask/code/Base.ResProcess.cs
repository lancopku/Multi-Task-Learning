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
    class resProcess
    {
        public static void resSummarize(string fn = "f2")
        {
            StreamReader sr = new StreamReader(Global.outDir + Global.fRawRes);
            string txt = sr.ReadToEnd();
            sr.Close();
            txt = txt.Replace("\r", "");
            string[] regions = txt.Split(Global.triLnEndAry, StringSplitOptions.RemoveEmptyEntries);
            StreamWriter sw = new StreamWriter(Global.outDir + Global.fResSum);
            foreach (string region in regions)
            {
                string[] blocks = region.Split(Global.biLnEndAry, StringSplitOptions.RemoveEmptyEntries);
                List<dMatrix> mList = new List<dMatrix>();
                foreach (string im in blocks)
                {
                    dMatrix m = new dMatrix(im);
                    mList.Add(m);
                }

                //get average
                dMatrix avgM = new dMatrix(mList[0]);
                avgM.set(0);
                foreach (dMatrix m in mList)
                    avgM.add(m);
                avgM.divide(mList.Count);
                //get devi
                dMatrix deviM = mathTool.getDeviation(mList);

                sw.WriteLine("%averaged values:");
                avgM.write(sw, fn);
                sw.WriteLine();
                sw.WriteLine("%deviations:");
                deviM.write(sw, fn);
                sw.WriteLine();
                sw.WriteLine("%avg & devi:");

                sMatrix sAvgM = new sMatrix(avgM, fn);
                sAvgM.add("$\\pm$");
                sMatrix sDeviM = new sMatrix(deviM, fn);
                sAvgM.add(sDeviM);
                sAvgM.write(sw);
                sw.WriteLine("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n");
            }
            sw.Close();
        }
        
        public static void write_multi()
        {
            if (Global.runMode.Contains("train"))//train
            {
                Global.swResRaw.WriteLine("%iter#, task#1-score(%), task#2-score(%), ..., task#n-score(%), averaged-score(%), time(sec), obj");
                dMatrix m = new dMatrix(Global.scoreTaskList_multi, true);

                int row, col;
                for (row = 0; row < m.R; row++)
                {
                    Global.swResRaw.Write((row + 1).ToString() + ",");

                    double avg = 0;
                    for (col = 0; col < m.C; col++)
                    {
                        avg += m[row, col];
                        Global.swResRaw.Write(m[row, col].ToString("f2") + ",");
                    }
                    avg /= m.C;
                    Global.swResRaw.Write(avg.ToString("f2") + ",");

                    double time = 0;
                    for (int k = 0; k <= row; k++)
                        time += Global.timeList_multi[k];
                    Global.swResRaw.Write(time.ToString("f2") + ",");

                    Global.swResRaw.Write(Global.errorList_multi[row].ToString("f2"));
                    Global.swResRaw.WriteLine();
                }
                Global.swResRaw.Flush();
            }
            else//test
            {
                Global.swResRaw.WriteLine("%task#1-score(%), task#2-score(%), ..., task#n-score(%), averaged-score(%)");
                double avg = 0;
                for (int i = 0; i < Global.nTask; i++)
                {
                    double score = Global.scoreTaskList_multi[i][0];
                    Global.swResRaw.Write(score.ToString("f2") + ",");
                    avg += score;
                }
                avg /= (double)Global.nTask;
                Global.swResRaw.Write(avg.ToString("f2"));
            }

            //clear
            for (int i = 0; i < Global.ttlIter; i++)
            {
                Global.timeList_multi[i] = 0;
                Global.errorList_multi[i] = 0;
            }
            for (int i = 0; i < Global.scoreTaskList_multi.Count; i++)
            {
                Global.scoreTaskList_multi[i].Clear();
            }
        }

    }
    
}
