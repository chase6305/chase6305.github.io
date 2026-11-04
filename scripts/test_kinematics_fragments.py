#!/usr/bin/env python3
"""Compile published local C++ wrist/limit fragments; independent DH oracle.
Does not validate the historical full inverse solver, collisions or hardware.
"""
from pathlib import Path
import math, random, subprocess, tempfile
ROOT = Path(__file__).resolve().parents[1]
ARTICLE = ROOT / 'content/posts/robotics/kinematics/six-dof-kinematics/index.md'

def multiply(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

def rotation(q):
    def rz(a):
        c,s=math.cos(a),math.sin(a)
        return [[c,-s,0],[s,c,0],[0,0,1]]
    rx=[[1,0,0],[0,0,-1],[0,1,0]]
    return multiply(multiply(multiply(multiply(rz(q[0]),rx),rz(q[1])),rx),rz(q[2]))

def main():
    text=ARTICLE.read_text()
    wrist=text[text.index('    // 本片段的腕部约定'):text.index('    // 6轴串联关节机器人有三种奇点')]
    limits=text[text.index('std::vector<double> Kinematics::lim'):text.index('\n\n\n/**',text.index('std::vector<double> Kinematics::lim'))]
    cpp='''#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
using namespace std;
const double M_PI_set=acos(-1.0), floatPrecision=1e-20;
struct Matrix { double v[9]; double operator()(int i,int j)const{return v[i*3+j];} };
struct Solution { double angles[6]{}; };
struct Limit { double minAngle=-20, maxAngle=20; };
struct Kinematics { static vector<double> lim(const double&,const double&,double); };
'''+limits+'''
int main(){ cout<<setprecision(17); char mode;
while(cin>>mode){ if(mode=='l'){double lo,hi,q;cin>>lo>>hi>>q; auto v=Kinematics::lim(hi,lo,q);cout<<v.size();for(double a:v)cout<<' '<<a;cout<<'\\n';}
else {Matrix R36;for(double& x:R36.v)cin>>x; double current[6]{};cin>>current[3]>>current[5];Solution sol_up,sol_down;Limit actuatorConfigType[6];
'''+wrist+'''
for(auto sol:{sol_up,sol_down}) for(int j=3;j<6;j++)cout<<sol.angles[j]<<' ';cout<<'\\n'; }} }
'''
    rng=random.Random(6305); queries=[]; expected=[]
    for b in [0,math.pi,-math.pi]+[rng.uniform(-3,3) for _ in range(250)]:
        for a,c in [(2.4,0),(-2.4,.7),(0,0),(1.2,-2.1)]:
            r=rotation([a,b,c]); current=[rng.uniform(-3,3),rng.uniform(-6,6)]
            queries.append('w '+' '.join(map(repr,[x for row in r for x in row]+current)))
            expected.append(('w',r))
    for lo,hi in [(0,1),(-math.pi,math.pi),(-8,8),(-2,15)]:
        for q in [lo,hi,lo+2*math.pi,hi+4*math.pi]+[rng.uniform(-25,25) for _ in range(250)]:
            values=sorted(q+2*math.pi*k for k in range(-20,21) if lo<=q+2*math.pi*k<hi)
            queries.append(f'l {lo!r} {hi!r} {q!r}');expected.append(('l',values))
    with tempfile.TemporaryDirectory(prefix='chase-wrist-') as tmp:
        p=Path(tmp);(p/'check.cpp').write_text(cpp)
        subprocess.run(['c++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-Wno-misleading-indentation',str(p/'check.cpp'),'-o',str(p/'check')],check=True)
        run=subprocess.run([str(p/'check')],input='\n'.join(queries)+'\n',text=True,capture_output=True,check=True,timeout=15)
    rows=run.stdout.splitlines();assert len(rows)==len(expected)
    maximum=0.0
    for line,(kind,target) in zip(rows,expected):
        values=list(map(float,line.split()))
        if kind=='l':
            actual=sorted(values[1:]);assert int(values[0])==len(actual)==len(target),(line,target)
            assert all(abs(a-b)<3e-14 for a,b in zip(actual,target)),(actual,target)
        else:
            assert len(values)==6
            for q in [values[:3],values[3:]]:
                actual=rotation(q);error=max(abs(actual[i][j]-target[i][j]) for i in range(3) for j in range(3));maximum=max(maximum,error)
                assert error<2e-13,(q,error,target)
    print(f'PASS {len(rows)} compiled fragment cases; wrist max matrix-entry error {maximum:.3g}; full IK not tested')

def check_seven():
    import re
    import numpy as np
    article = ROOT / 'content/posts/robotics/kinematics/seven-dof-kinematics/index.md'
    source = re.search(r'```python\n(.*?)\n```', article.read_text(), re.S).group(1)
    namespace = {'__name__': 'kinematics_fragment_test'}
    exec(compile(source, str(article), 'exec'), namespace)
    solver = namespace['SRSKinSolver']()
    def fk(q):
        # Independent scalar DH chain; no calls to the article's FK implementation.
        r = [[1.,0,0],[0,1.,0],[0,0,1.]]
        position = [0.,0,0]
        for angle, offset, twist in zip(q, [.34,0,.4,0,.4,0,.126],
                [-math.pi/2,math.pi/2,math.pi/2,-math.pi/2,-math.pi/2,math.pi/2,0]):
            position = [position[k]+r[k][2]*offset for k in range(3)]
            c,s = math.cos(angle),math.sin(angle)
            ct,st = math.cos(twist),math.sin(twist)
            r = multiply(multiply(r, [[c,-s,0],[s,c,0],[0,0,1]]),
                         [[1,0,0],[0,ct,-st],[0,st,ct]])
        return np.array([r[k]+[position[k]] for k in range(3)]+[[0.,0,0,1.]])
    maximum = 0.; count = 0
    for radial in [0.,1e-10,1e-8,2e-7,8e-7,9.99e-7,1.01e-6,1e-4]:
        for azimuth in [0.,.7,math.pi,-2.1]:
            target = np.eye(4)
            target[:3,3] = [radial*math.cos(azimuth),radial*math.sin(azimuth),1.066]
            for branch in range(8):
                q,_,_ = solver.inverse_kinematics(target,.3,branch)
                error = float(np.max(np.abs(fk(q)-target)))
                maximum = max(maximum,error);count += 1
                assert error < 2e-12, (radial,azimuth,branch,error)
    print(f'PASS {count} seven-axis near-vertical branches; max transform-entry error {maximum:.3g}')

if __name__=='__main__':
    import sys
    main()
    if '--numpy' in sys.argv:check_seven()
