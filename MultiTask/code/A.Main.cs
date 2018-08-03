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
using System.Diagnostics; 

namespace Program
{
    class MainClass
    {
        static void Main(string[] args)
        {
            Stopwatch timer = new Stopwatch();
            timer.Start();

            int flag = readCommand(args);
            if (flag == 1)
            {
                return;
            }
            else if (flag == 2)
            {
                Console.WriteLine("commands invalid...type 'help' for information on commands.");
                return;
            }

            Global.globalCheck();
            directoryCheck();

            Global.timeList_multi = new List<double>(new double[Global.ttlIter]);
            Global.errorList_multi = new List<double>(new double[Global.ttlIter]);
            for (int i = 0; i < Global.nTask; i++)
            {
                Global.scoreTaskList_multi.Add(new List<double>());
            }

            Global.swLog = new StreamWriter(Global.outDir + Global.fLog);
            Global.swResRaw = new StreamWriter(Global.outDir + Global.fRawRes);
            Global.swSimi = new StreamWriter(Global.outDir + Global.fSimi);
            if (Global.runMode.Contains("mt"))
            {
                for (int i = 0; i < Global.nTask; i++)
                {
                    Global.swOutputList.Add(new StreamWriter(Global.outDir+i.ToString()+Global.fOutput));
                }
            }
            else  
                Global.swOutput = new StreamWriter(Global.outDir + Global.fOutput);
            
            Global.swLog.WriteLine("exe command:");
            string cmd = "";
            foreach (string im in args)
                cmd += im + " ";
            Global.swLog.WriteLine(cmd);
            Global.printGlobals();

            //run
            if (Global.runMode.Contains("mt"))//multi-task
                multiTask();
            
            else throw new Exception("error");

            timer.Stop();
            double time = timer.ElapsedMilliseconds / 1000.0;
            Global.swLog.WriteLine("\ndone. used time (seconds): " + time.ToString());
            Console.WriteLine("\ndone. used time (seconds): " + time.ToString());

            Global.swLog.Close();
            Global.swResRaw.Close();
            Global.swSimi.Close();
            if (Global.runMode.Contains("mt"))
            {
                foreach (StreamWriter sw in Global.swOutputList)
                    sw.Close();
            }
            else
                Global.swOutput.Close();

            //get avg, devi
            resProcess.resSummarize("f2");
            //Console.Read();
        }

        static void multiTask()
        {
            //train
            if (Global.runMode.Contains("train"))
            {
                //load data
                List<dataSet> XList = new List<dataSet>();
                List<dataSet> XXList = new List<dataSet>();
                dataSet X = new dataSet();
                loadData_multi(XList, X, XXList);

                toolbox toolbox;

                //single-task training in multi-task framework: each task has its own independent train & test data
                if (Global.mt_singleTrain)
                {
                    foreach (double r in Global.regList)//experiments for each different regularizer value
                    {
                        Global.swResRaw.WriteLine("\n%single-task! r: {0}", r);
                        Console.WriteLine("\nsingle-task! r: {0}", r);

                        for (int i = 0; i < Global.nTask; i++)
                        {
                            Global.swLog.WriteLine("\nsingle-task! #task, r: " + (i + 1).ToString() + "," + r.ToString());
                            Console.WriteLine("\nsingle-task! #task, r: " + (i + 1).ToString() + "," + r.ToString());
                            Global.reg = r;
                            dataSet Xi = XList[i];
                            toolbox = new toolbox(Xi);
                            train_multi_single(XXList, toolbox, i);
                        }
                        resProcess.write_multi();
                    }
                    Global.swResRaw.WriteLine();
                }

                //merged training in multi-task framework: merge all training data to train a unified model
                if (Global.mt_mergeTrain)
                {
                    foreach (double r in Global.regList)//experiments for each different regularizer value
                    {
                        Global.reg = r;
                        Global.swLog.WriteLine("\nmerged-task! r: " + r.ToString());
                        Console.WriteLine("\nmerged-task! r: " + r.ToString());
                        Global.swResRaw.WriteLine("\n%merged-task! r: " + r.ToString());
                        toolbox = new toolbox(X);
                        train_multi_merge(XXList, toolbox);

                        resProcess.write_multi();
                    }
                    Global.swResRaw.WriteLine();
                }

                //multi-task learning
                if (Global.mt_mtTrain)
                {
                    foreach (double r in Global.regList)//experiments for each different regularizer value
                    {
                        Global.reg = r;
                        foreach (double cFactor in Global.cFactors)//experiments for each different C value (see Eq. 18 & 19 of [Sun+ TKDE 2013] for the definition of C)
                        {
                            Global.C = cFactor;
                            Global.swLog.WriteLine("\n%multi-task! reg, rate0, C, kernel: {0},{1},{2},{3}", Global.reg, Global.rate0, Global.C, Global.simiMode);
                            Global.swSimi.WriteLine("\n%multi-task! reg, rate0, C, kernel: {0},{1},{2},{3}", Global.reg, Global.rate0, Global.C, Global.simiMode);
                            Global.swResRaw.WriteLine("\n%multi-task! reg, rate0, C, kernel: {0},{1},{2},{3}", Global.reg, Global.rate0, Global.C, Global.simiMode);
                            Console.WriteLine("\nmulti-task! reg, rate0, C, kernel: {0},{1},{2},{3}", Global.reg, Global.rate0, Global.C, Global.simiMode);
                            toolbox = new toolbox(X, XList);
                            train_multi_mtl(XXList, toolbox);

                            resProcess.write_multi();
                        }
                    }
                    Global.swResRaw.WriteLine();
                }
            }
            else if (Global.runMode.Contains("test1"))//normal test
            {
                //load data
                List<dataSet> XList = new List<dataSet>();
                List<dataSet> XXList = new List<dataSet>();
                dataSet X = new dataSet();
                loadData_multi(XList, X, XXList);
                //load model etc.
                toolbox tb = new toolbox(X, XList, false);

                if (Global.mt_mergeTrain)//multi_merge
                {
                    List<double> scoreList = tb.test_multi_merge(XXList, 0, Global.swOutputList);
                    for (int i = 0; i < Global.nTask; i++)
                        Global.scoreTaskList_multi[i].Add(scoreList[i]);
                    resProcess.write_multi();
                }
                else//multi_single or multi_mtl: they have the same testing schema
                {
                    List<double> scoreList = tb.test_multi_mtl(XXList, 0, Global.swOutputList);
                    for (int i = 0; i < Global.nTask; i++)
                        Global.scoreTaskList_multi[i].Add(scoreList[i]);
                    resProcess.write_multi();
                }
            }
            else if (Global.runMode.Contains("test2"))//for multi_mtl: test a new task via choosing the most similar model
            {
                //load data
                List<dataSet> XList = new List<dataSet>();
                List<dataSet> XXList = new List<dataSet>();
                dataSet X = new dataSet();
                loadData_multi(XList, X, XXList);
                //get vectors 
                List<List<double>> vecList = new List<List<double>>();
                foreach (dataSet Xi in XList)
                {
                    List<double> vec = getVecFromX(Xi);
                    vecList.Add(vec);
                }
                //load model & test
                toolbox tb = new toolbox(X, XList, false);
                List<double> scoreList = tb.test2_multi_mtl(vecList, XXList, 0, Global.swOutputList);
                for (int i = 0; i < Global.nTask; i++)
                    Global.scoreTaskList_multi[i].Add(scoreList[i]);
                resProcess.write_multi();
            }
            else if (Global.runMode.Contains("test3"))//for multi_mtl: test a new task via voted-test based on all models, i.e., the OMT-SBD method described in Section 4.4 of [Sun+ TKDE 2013]
            {
                //load data
                List<dataSet> XList = new List<dataSet>();
                List<dataSet> XXList = new List<dataSet>();
                dataSet X = new dataSet();
                loadData_multi(XList, X, XXList);
                //get vectors 
                List<List<double>> vecList = new List<List<double>>();
                foreach (dataSet Xi in XList)
                {
                    List<double> vec = getVecFromX(Xi);
                    vecList.Add(vec);
                }
                //load model & test
                toolbox tb = new toolbox(X, XList, false);
                List<double> scoreList = tb.test3_multi_mtl(vecList, XXList, 0, Global.swOutputList);
                for (int i = 0; i < Global.nTask; i++)
                    Global.scoreTaskList_multi[i].Add(scoreList[i]);
                resProcess.write_multi();
            }
            else
                throw new Exception("error");
        }

        //single task learning in multi-task framework
        public static void train_multi_single(List<dataSet> XXList, toolbox tb, int taskID)
        {
            Global.reinitGlobal();

            if (Global.optim.Contains("bfgs"))
            {
                Global.bfgsTb = tb;//to fix
                Global.bfgsXXList = XXList;
                Global.bfgsTestMode = "mt.single";
                Global.bfgsTaskID = taskID;
            }

            for (double i = 0; i < Global.ttlIter; i++)
            {
                Global.glbIter++;
                Stopwatch timer = new Stopwatch();
                timer.Start();

                double error = tb.train_single();

                timer.Stop();
                double time = timer.ElapsedMilliseconds / 1000.0;
                Global.swLog.WriteLine("Training used time (second): " + time.ToString());

                //evaluate	
                if (!Global.optim.Contains("bfgs"))//test is already done in bfgs training
                {
                    List<double> scoreList = tb.test_single(XXList[taskID], i, Global.swOutputList[taskID]);
                    double score = scoreList[0];
                    Global.scoreTaskList_multi[taskID].Add(score);
                    Global.timeList_multi[Global.glbIter - 1] += time;
                    Global.errorList_multi[Global.glbIter - 1] += error;
                }
                if (Global.diff < Global.convergeTol)
                    break;
            }

            //save model
            if (Global.save == 1)
            {
                tb.Model.save(Global.modelDir + taskID.ToString() + Global.fModel);
            }
        }

        //merged learning in multi-task framework
        public static void train_multi_merge(List<dataSet> XXList, toolbox tb)
        {
            Global.reinitGlobal();

            if (Global.optim.Contains("bfgs"))
            {
                Global.bfgsTb = tb;
                Global.bfgsXXList = XXList;
                Global.bfgsTestMode = "mt.merge";
            }

            for (double i = 0; i < Global.ttlIter; i++)
            {
                Global.glbIter++;
                Stopwatch timer = new Stopwatch();
                timer.Start();

                double error = tb.train_single();

                timer.Stop();
                double time = timer.ElapsedMilliseconds / 1000.0;
                Global.swLog.WriteLine("Training used time (second): " + time.ToString());

                //evaluate	
                if (!Global.optim.Contains("bfgs"))//test is already done in bfgs training
                {
                    List<double> scoreList = tb.test_multi_merge(XXList, i, Global.swOutputList);
                    for (int k = 0; k < Global.nTask; k++)
                        Global.scoreTaskList_multi[k].Add(scoreList[k]);
                    Global.timeList_multi[Global.glbIter - 1] += time;
                    Global.errorList_multi[Global.glbIter - 1] += error;
                }
                if (Global.diff < Global.convergeTol)
                    break;
            }

            //save model
            if (Global.save == 1)
            {
                tb.Model.save(Global.modelDir + Global.fModel);
            }
        }

        //multi-task learning
        public static void train_multi_mtl(List<dataSet> XXList, toolbox tb)
        {
            Global.reinitGlobal();

            for (double iter = 0; iter < Global.ttlIter; iter++)
            {
                Global.glbIter++;
                Stopwatch timer = new Stopwatch();
                timer.Start();

                double error = tb.train_multi();

                timer.Stop();
                double time = timer.ElapsedMilliseconds / 1000.0;
                Global.swLog.WriteLine("Training used time (second): " + time.ToString());

                //evaluate	
                List<double> scoreList = tb.test_multi_mtl(XXList, iter, Global.swOutputList);
                for (int i = 0; i < Global.nTask; i++)
                    Global.scoreTaskList_multi[i].Add(scoreList[i]);
                Global.timeList_multi[Global.glbIter - 1] += time;
                Global.errorList_multi[Global.glbIter - 1] += error;

                if (iter > 30 && Global.diff > 0 && Global.diff < Global.convergeTol)
                    break;
            }

            //save model
            if (Global.save == 1)
            {
                for (int i = 0; i < Global.nTask; i++)
                {
                    tb.ModelList[i].save(Global.modelDir + i.ToString() + Global.fModel);
                }
            }
        }

        public static void dataSizeScale(dataSet X)
        {
            dataSet XX = new dataSet();
            XX.setDataInfo(X);
            foreach (dataSeq im in X)
                XX.Add(im);
            X.Clear();

            int n = (int)(XX.Count * Global.trainSizeScale);
            for (int i = 0; i < n; i++)
            {
                int j = i;
                if (j > XX.Count - 1)
                    j %= XX.Count - 1;
                X.Add(XX[j]);
            }
            X.setDataInfo(XX);
        }

        public static void loadData_multi(List<dataSet> XList, dataSet X, List<dataSet> XXList)
        {
            XList.Clear();
            XXList.Clear();
            //load train data
            baseHashSet<int> checkSet = new baseHashSet<int>();
            for (int i = 0; i < Global.nTask; i++)
            {
                string dat_i = i.ToString() + Global.fFeatureTrain;
                string tag_i = i.ToString() + Global.fGoldTrain;
                dataSet Xi = new dataSet(dat_i, tag_i);
                dataSizeScale(Xi);
                checkSet.Add(Xi.NFeatureTemp);
                XList.Add(Xi);
            }
            if (checkSet.Count > 1)
                throw new Exception("inconsistent features among multi tasks!");

            //make nTag consistent among different tasks
            int maxNTag = 0;
            foreach (dataSet Xi in XList)
            {
                if (maxNTag < Xi.NTag)
                    maxNTag = Xi.NTag;
            }
            for (int i = 0; i < Global.nTask; i++)
            {
                XList[i].NTag = maxNTag;
            }

            //add to merged data
            X.NTag = XList[0].NTag;
            X.NFeatureTemp = XList[0].NFeatureTemp;
            foreach (dataSet Xi in XList)
                foreach (dataSeq im in Xi)
                    X.Add(im);
            Global.swLog.WriteLine("data sizes (1, ..., T):");
            for (int i = 0; i < Global.nTask; i++)
            {
                dataSet Xi = XList[i];
                Global.swLog.WriteLine(" " + Xi.Count.ToString());
            }
            Global.swLog.WriteLine();

            //load test data 
            for (int i = 0; i < Global.nTask; i++)
            {
                string dat_i = i.ToString() + Global.fFeatureTest;
                string tag_i = i.ToString() + Global.fGoldTest;
                dataSet Xtest = new dataSet(dat_i, tag_i);
                XXList.Add(Xtest);
            }
            for (int i = 0; i < Global.nTask; i++)
                XXList[i].NTag = maxNTag;
        }

        public static void loadTestData_multi(List<dataSet> XXList)
        {
            XXList.Clear();
            //load test data 
            Global.swLog.WriteLine("test data sizes (1, ..., T):");
            for (int i = 0; i < Global.nTask; i++)
            {
                string dat_i = i.ToString() + Global.fFeatureTest;
                string tag_i = i.ToString() + Global.fGoldTest;
                dataSet Xtest = new dataSet(dat_i, tag_i);
                Global.swLog.WriteLine(" " + Xtest.Count.ToString());
                XXList.Add(Xtest);
            }
            Global.swLog.WriteLine();
        }

        public static List<double> getVecFromX(dataSet X)
        {
            List<double> vec = new List<double>(new double[X.NFeatureTemp]);
            int nodeCount = 0;
            foreach (dataSeq x in X)
            {
                List<List<featureTemp>> featureTemps = x.getFeatureTemp();
                foreach (List<featureTemp> ftList in featureTemps)
                {
                    foreach (featureTemp ft in ftList)
                    {
                        vec[ft.id] += ft.val;
                    }
                    nodeCount++;
                }
            }
            listTool.listMultiply(ref vec, 1.0 / (double)nodeCount);
            return vec;
        }

        static void directoryCheck()
        {
            if (!Directory.Exists(Global.modelDir))
                Directory.CreateDirectory(Global.modelDir);
            if(Global.runMode.Contains("train"))
                fileTool.removeFile(Global.modelDir);
            Global.outDir = Directory.GetCurrentDirectory() + "/" + Global.outFolder + "/";
            if (!Directory.Exists(Global.outDir))
                Directory.CreateDirectory(Global.outDir);
            fileTool.removeFile(Global.outDir);
        }

        static int readCommand(string[] args)
        {
            foreach (string arg in args)
            {
                if (arg == "help")
                {
                    helpCommand();
                    return 1;
                }
                string[] ary = arg.Split(Global.colonAry, StringSplitOptions.RemoveEmptyEntries);
                if (ary.Length != 2)
                    return 2;
                string opt = ary[0], val = ary[1];

                switch (opt)
                {
                    case "m":
                        Global.runMode = val;
                        break;
                    case "o":
                        Global.optim = val;
                        break;
                    case "a":
                        Global.rate0 = double.Parse(val);
                        break;
                    case "r":
                        Global.regList.Clear();
                        string[] regAry = val.Split(Global.commaAry, StringSplitOptions.RemoveEmptyEntries);
                        foreach (string im in regAry)
                        {
                            Global.regList.Add(double.Parse(im));
                        }
                        break;
                    case "d":
                        Global.random = int.Parse(val);
                        break;
                    case "e":
                        Global.evalMetric = val;
                        break;
                    case "t":
                        Global.taskBasedChunkInfo = val;
                        break;
                    case "ss":
                        Global.trainSizeScale = double.Parse(val);
                        break;
                    case "i":
                        Global.ttlIter = int.Parse(val);
                        break;
                    case "s":
                        if (val == "1")
                            Global.save = 1;
                        else
                            Global.save = 0;
                        break;
                    case "of":
                        Global.outFolder = val;
                        break;
                    default:
                        return 2;
                }
            }
            return 0;//success
        }

        static void helpCommand()
        {
            Console.WriteLine("'option1:value1  option2:value2 ...' for setting values to options.");
            Console.WriteLine("'m' for runMode. Default: {0}", Global.runMode);
            Console.WriteLine("'o' for optim. Default: {0}", Global.optim);
            Console.WriteLine("'a' for rate0. Default: {0}", Global.rate0);
            Console.WriteLine("'r' for regs. E.g., 'r:1,2,3'. Default: {0}", Global.regList[0]);
            Console.WriteLine("'d' for random. Default: {0}", Global.random);
            Console.WriteLine("'e' for evalMetric. Default: {0}", Global.evalMetric);
            Console.WriteLine("'t' for taskBasedChunkInfo. Default: {0}", Global.taskBasedChunkInfo);
            Console.WriteLine("'ss' for trainSizeScale. Default: {0}", Global.trainSizeScale);
            Console.WriteLine("'i' for ttlIter. Default: {0}", Global.ttlIter);
            Console.WriteLine("'s' for save. Default: {0}", Global.save);
            Console.WriteLine("'of' for outFolder. Default: {0}", Global.outFolder);
        }

    }
}
