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
    class Global
    {
        //default values of the commands
        public static string runMode = "mt.train.fast";//mt.train, mt.train.fast, mt.test1, mt.test2, mt.test3
        public static string optim = "sgd";//sgd, sgder, bfgs
        public static double rate0 = 0.1;//init value of decay rate in SGD 
        static double[] regs = { 1};
        public static List<double> regList = new List<double>(regs);
        public static int random = 0;//0 for 0-initialization of model weights, 1 for random init of model weights
        public static string evalMetric = "tok.acc";//tok.acc, str.acc, f1
        public static string taskBasedChunkInfo = "";//for f1 score: np.chunk, bio.ner, wd.seg
        public static double trainSizeScale = 1;//for scaling the size of training data
        public static int ttlIter = 100;//# of training iterations
        public static string outFolder = "out";
        public static int save = 1;//save model file
        //multi-task
        public static bool mt_singleTrain = false;
        public static bool mt_mergeTrain = false;
        public static bool mt_mtTrain = true;
        public static int nTask = 4;
        public static double[] cFactors = { 10 };//kern/=cFactor, #task is not involved
        public static string simiMode = "cov";//cov (covariance), poly (polynomial), rbf (Gaussian RBF)
        public static double sampleFactor = 3;
        public const double simiUpdateIter = 3;
        //general
        public static double convergeTol = -1e10;
        public static bool debugMode = false;
        public static double decayFactor = 0.94;//decay factor in SGD training
        public static int scalarResetStep = 1000;
        //LBFGS
        public static int mBFGS = 10;
        public static bool wolfe = true;
 
        //global variables
        public static baseHashMap<string, string> chunkTagMap = new baseHashMap<string, string>();
        public static List<double> timeList_multi;
        public static List<double> errorList_multi;
        public static List<List<double>> scoreTaskList_multi = new List<List<double>>();
        public static string bfgsTestMode;
        public static int bfgsTaskID;
        public static toolbox bfgsTb;
        public static dataSet bfgsXX;
        public static List<dataSet> bfgsXXList;
        public static string outDir = "";
        public static double C = 1;
        public static double reg = 1;
        public static int glbIter = 0;
        public static double diff = double.MaxValue;
        public static int countWithIter = 0;
        public static StreamWriter swLog;
        public static StreamWriter swResRaw;
        public static StreamWriter swSimi;
        public static StreamWriter swOutput;
        public static List<StreamWriter> swOutputList = new List<StreamWriter>();
        public static char[] lineEndAry = { '\n' };
        public static string[] biLnEndAry = { "\n\n" };
        public static string[] triLnEndAry = { "\n\n\n" };
        public static char[] barAry = { '-' };
        public static char[] underlnAry = { '_' };
        public static char[] commaAry = { ',' };
        public static char[] tabAry = { '\t' };
        public static char[] vertiBarAry = { '|' };
        public static char[] colonAry = { ':' };
        public static char[] blankAry = { ' ' };
        public static char[] starAry = { '*' };
        public static char[] slashAry = { '/' };
        public const string modelDir = "model/";
        public const string fLog = "trainLog.txt";
        public const string fResSum = "summarizeResult.txt";
        public const string fRawRes = "rawResult.txt";
        public const string fSimi = "similarity.txt";
        public const string fFeatureTrain = "ftrain.txt";
        public const string fGoldTrain = "gtrain.txt";
        public const string fFeatureTest = "ftest.txt";
        public const string fGoldTest = "gtest.txt";
        public const string fModel = "model.txt";
        public const string fOutput = "taskOutput.txt";

        public static void reinitGlobal()
        {
            diff = double.MaxValue;
            countWithIter = 0;
            glbIter = 0;
        }

        public static void globalCheck()
        {
            if (trainSizeScale != 1)
                Console.WriteLine("Note: trainSizeScale!");
        }

        public static void printGlobals()
        {
            swLog.WriteLine("runMode: {0}", Global.runMode);
            swLog.WriteLine("optim: {0}", Global.optim);
            swLog.WriteLine("rate0: {0}", Global.rate0);
            swLog.WriteLine("regs: {0}", Global.regList[0]);
            swLog.WriteLine("random: {0}", Global.random);
            swLog.WriteLine("evalMetric: {0}", Global.evalMetric);
            swLog.WriteLine("taskBasedChunkInfo: {0}", Global.taskBasedChunkInfo);
            swLog.WriteLine("trainSizeScale: {0}", Global.trainSizeScale);
            swLog.WriteLine("ttlIter: {0}", Global.ttlIter);
            swLog.WriteLine("outFolder: {0}", Global.outFolder);
        }

        //the system must know the B (begin-chunk), I (in-chunk), O (out-chunk) information for computing f-score
        //since such BIO information is task-dependent, it should be explicitly coded here
        static void getChunkTagMap()
        {
            chunkTagMap.Clear();

            /*
            Noun phrase chunk task (Sun et al. COLING 2008)'s BIO information
            O    0
            I-NP    1
            B-NP    2
            */
            if (Global.taskBasedChunkInfo == "np.chunk")
            {
                chunkTagMap["0"] = "O";
                chunkTagMap["1"] = "I";
                chunkTagMap["2"] = "B";
            }
            /*
            biomedical named entity recognition task (Sun et al. IJCAI 2009)'s BIO information
            I-RNA    0
            O    1        
            B-protein    2
            B-RNA    3
            B-cell_type    4
            B-cell_line    5
            B-DNA    6
            I-protein    7
            I-DNA    8
            I-cell_type    9
            I-cell_line    10
            */
            else if (Global.taskBasedChunkInfo == "bio.ner")
            {
                chunkTagMap["0"] = "I";
                chunkTagMap["1"] = "O";
                chunkTagMap["2"] = "B1";
                chunkTagMap["3"] = "B2";
                chunkTagMap["4"] = "B3";
                chunkTagMap["5"] = "B4";
                chunkTagMap["6"] = "B5";
                chunkTagMap["7"] = "I";
                chunkTagMap["8"] = "I";
                chunkTagMap["9"] = "I";
                chunkTagMap["10"] = "I";
            }
            /*
            Chinese word segmentation task (Sun et al. ACL 2012)'s BIO information
            B    0
            E    1
            I    2
            */
            else if (Global.taskBasedChunkInfo == "wd.seg")
            {
                chunkTagMap["0"] = "B";
                chunkTagMap["1"] = "I";
                chunkTagMap["2"] = "I";
            }
            else
                throw new Exception("error");
        }

    }

}
