using System;
using System.Diagnostics;
using System.IO;

namespace run_python
{
    class Program
    {
        public static void run_cmd(string script, string args)
        {
            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = "C:/Users/tonyh/AppData/Local/Programs/Python/Python37/python.exe";
            start.Arguments = string.Format("\"{0}\" \"{1}\"", script, args);
            start.UseShellExecute = false;
            start.CreateNoWindow = true;
            start.RedirectStandardOutput = true;
            start.RedirectStandardError = true;
            using (Process process = Process.Start(start))
            {
                using (StreamReader reader = process.StandardOutput)
                {
                    string stderr = process.StandardError.ReadToEnd();
                    string result = reader.ReadToEnd();
                    Console.WriteLine(result);
                }
            }
        }
        static void Main(string[] args)
        {
            string scipt = @"c:/Users/tonyh/Desktop/python_server_and_client/client.py";
            run_cmd(scipt, "hello my friend");
        }
    }
}
