// main101.cc is a part of the PYTHIA event generator.
// Copyright (C) 2025 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Keywords: basic usage; charged multiplicity

// This is a simple test program. It fits on one slide in a talk.
// It studies the charged multiplicity distribution at the LHC.


#include "Pythia8/Pythia.h"
#include <fstream>
#include <string>
using namespace Pythia8;
using namespace std;

int main() {
    // 总事件数和分文件设置
    int nTotalEvents = 10000000;
    int nSplits = 50;
    int nEventsPerFile = nTotalEvents / nSplits;

    // 初始化 Pythia
    Pythia pythia;
    pythia.readString("Beams:idA = 2212");   // p
    pythia.readString("Beams:idB = 2212");   // p
    pythia.readString("Beams:eCM = 200.");   // √s = 200 GeV

    // Minimum-bias 设置
    pythia.readString("SoftQCD:inelastic = on");         // 非弹性事件
    pythia.readString("SoftQCD:nonDiffractive = on");    // 非衍射事件
    pythia.readString("SoftQCD:singleDiffractive = off"); // 可选，关闭单衍射
    pythia.readString("SoftQCD:doubleDiffractive = off"); // 可选，关闭双衍射

    // Tune 保持原样
    pythia.readString("Tune:pp = 14");  // Monash 2013 tune

    if (!pythia.init()) return 1;

    int globalEventId = 0;

    // 循环生成 nSplits 个文件
    for (int split = 0; split < nSplits; ++split) {
        string filename = "Mevents_" + to_string(split) + ".txt";
        ofstream fout(filename);

        // 生成该文件中的事件
        for (int iEvent = 0; iEvent < nEventsPerFile; ++iEvent) {
            if (!pythia.next()) continue;
            globalEventId++;

            int particleId = 0;
            for (int i = 0; i < pythia.event.size(); ++i) {
                // 只保留末态 + 带电粒子
                if (!pythia.event[i].isFinal()) continue;
                if (!pythia.event[i].isCharged()) continue;

                particleId++;
                fout << globalEventId << " "                // event_id
                     << particleId   << " "                // particle_id
                     << pythia.event[i].id() << " "        // particle_type (PDG ID)
                     << pythia.event[i].px() << " "        // px
                     << pythia.event[i].py() << " "        // py
                     << pythia.event[i].pz() << "\n";      // pz
            }
        }

        fout.close();
        cout << "Finished file " << split+1 << "/" << nSplits
             << " -> " << filename << endl;
    }

    pythia.stat();
    return 0;
}
