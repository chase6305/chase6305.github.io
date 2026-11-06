#!/usr/bin/env python3
"""Constructed control/statistical counterexamples for a robot-learning review.

Standard library only. No robot commands, clinical conclusions, or SureSim
implementation. PPI arrays represent fixed-policy IID bounded observations.
"""
import argparse
from itertools import product
import math


def average(values):
    values = list(values)
    if not values:
        raise ValueError("nonempty data required")
    return sum(values) / len(values)


def bounded(values):
    values = list(values)
    if not values or any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError("expected nonempty observations in [0,1]")
    return values


def ppi_pooled(real, paired_sim, additional_sim):
    real, paired_sim, additional_sim = map(bounded, (real, paired_sim, additional_sim))
    if len(real) != len(paired_sim):
        raise ValueError("paired observations must have equal lengths")
    return average([y-f for y, f in zip(real, paired_sim)]) + average(paired_sim+additional_sim)


def pooled_variance(population_real, population_sim, paired_count, additional_count):
    """Exact IID variance given a known finite joint population, with replacement."""
    y, f = bounded(population_real), bounded(population_sim)
    if len(y) != len(f) or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in (paired_count, additional_count)):
        raise ValueError("joint population and positive integer sample counts required")
    my, mf = average(y), average(f)
    vy = average((x-my)**2 for x in y)
    vf = average((x-mf)**2 for x in f)
    cov = average((a-my)*(b-mf) for a, b in zip(y, f))
    n, extra = paired_count, additional_count
    weight = extra / (n+extra)
    return (vy+weight**2*vf-2*weight*cov)/n+weight**2*vf/extra


def zero_failure_upper(trials, alpha=.05):
    """Exact one-sided binomial upper bound after zero failures; fixed IID trials."""
    if not isinstance(trials, int) or isinstance(trials, bool) or trials < 1 or not 0 < alpha < 1:
        raise ValueError("positive integer trials and 0<alpha<1 required")
    return -math.expm1(math.log(alpha)/trials)


def semidefinite_state(t, x0=1., y0=1.):
    if not math.isfinite(t) or t < 0:
        raise ValueError("finite nonnegative time required")
    x, y = x0*math.exp(-t), y0
    return x, y, x*x+y*y, -2*x*x


def scalar_cbf(x, nominal, alpha=10., boundary=.3, actuator=5.):
    """Project a scalar nominal velocity onto CBF and actuator constraints.

    Model: xdot=u, safe set x<=boundary. None means the inequalities are
    infeasible; this helper never invents a feasible control in that case.
    """
    if not all(math.isfinite(v) for v in (x,nominal,alpha,boundary,actuator)) or alpha <= 0 or actuator <= 0:
        raise ValueError("finite inputs and positive alpha/actuator required")
    upper = min(actuator, alpha*(boundary-x))
    if upper < -actuator:
        return None
    return max(-actuator,min(nominal,upper))


def held_velocity(x, nominal, hold, boundary=.3, actuator=5., disturbance_upper=0.):
    """Exact robust held-input limit for xdot=u+d, x<=boundary.

    Initial x must be inside the set, u is constant for `hold` seconds,
    d(t)<=disturbance_upper, and there is no delay or state error. A lower
    disturbance bound is unnecessary for this single upper state constraint.
    None means no actuator-feasible command meets this sufficient bound.
    """
    values = (x, nominal, hold, boundary, actuator, disturbance_upper)
    if (not all(math.isfinite(v) for v in values) or hold <= 0 or actuator <= 0
            or disturbance_upper < 0 or x > boundary):
        raise ValueError("finite inputs, initially safe x, and positive hold/actuator required")
    upper = min(actuator, (boundary-x)/hold-disturbance_upper)
    if upper < -actuator:
        return None
    return max(-actuator, min(nominal, upper))


def continuous_boundary(t, alpha=10., boundary=.3):
    return boundary*(-math.expm1(-alpha*t))


def check():
    from decimal import Decimal, localcontext
    import random
    population = [0.,0.,1.,1.]
    for predicted in [population,[1-x for x in population],[.2+.6*x for x in population],[.7]*4]:
        estimates=[]
        for indices in product(range(4),repeat=4):
            a,b,c,d=indices
            estimates.append(ppi_pooled([population[a],population[b]],[predicted[a],predicted[b]],[predicted[c],predicted[d]]))
        mean=average(estimates)
        var=average((x-mean)**2 for x in estimates)
        assert abs(mean-.5)<1e-13
        assert abs(var-pooled_variance(population,predicted,2,2))<1e-13
    assert abs(pooled_variance(population,population,2,2)-.0625)<1e-14
    assert abs(pooled_variance(population,[1-x for x in population],2,2)-.3125)<1e-14
    assert .3125 > .25/2
    for n in (1,20,100,1000,10**8):
        upper=zero_failure_upper(n)
        assert 0<upper<1
        if n<=1000:
            assert abs((1-upper)**n-.05)<1e-13
        with localcontext() as context:
            context.prec=60
            expected=1-(Decimal('.05').ln()/Decimal(n)).exp()
        assert abs(upper-float(expected))<2e-16
    assert zero_failure_upper(20)>zero_failure_upper(100)>zero_failure_upper(1000)
    for target in (.001,.0001):
        count = math.ceil(math.log(.05)/math.log1p(-target))
        assert zero_failure_upper(count) <= target < zero_failure_upper(count-1)
    assert math.ceil(math.log(.05)/math.log1p(-.001)) == 2995
    x,y,v,dv=semidefinite_state(30)
    assert abs(x)<1e-12 and y==1 and v>=1 and dv<=0
    rng=random.Random(81804)
    for _ in range(300):
        x=rng.uniform(-1,.3)
        nominal=rng.uniform(-10,10)
        u=scalar_cbf(x,nominal)
        assert u is not None and -5<=u<=5
        assert u <= 10*(.3-x)+1e-14
        # With dt*alpha<=1, the scalar forward step preserves h>=0 here.
        assert x+.05*u<=.3+1e-14
    assert scalar_cbf(.8,1.,actuator=1.) is None
    u=scalar_cbf(0.,3.)
    assert u==3. and .2*u>.3
    assert continuous_boundary(.2)<.3
    assert abs(held_velocity(0.,3.,.2) - 1.5) < 1e-14
    assert abs(held_velocity(0.,3.,.2,disturbance_upper=.2)-1.3) < 1e-14
    assert held_velocity(.3,1.,.2,actuator=.1,disturbance_upper=.2) is None
    # Exact interval maximum is at an endpoint for constant worst-case drift.
    # Piecewise disturbances below the bound cannot exceed that upper envelope.
    for _ in range(300):
        x = rng.uniform(-1.,.3)
        hold = rng.uniform(.001,1.)
        disturbance_upper = rng.uniform(0.,2.)
        u = held_velocity(x,rng.uniform(-10.,10.),hold,
                          actuator=2.,disturbance_upper=disturbance_upper)
        assert u is not None and -2. <= u <= 2.
        worst_endpoint = x + hold*(u+disturbance_upper)
        assert max(x,worst_endpoint) <= .3 + 2e-14
        state = x
        for _ in range(20):
            disturbance = rng.uniform(-3.,disturbance_upper)
            state += hold/20*(u+disturbance)
            assert state <= .3 + 2e-14
    for t in (.01,.03,.1,.18):
        dt=1e-6
        derivative=(continuous_boundary(t+dt)-continuous_boundary(t-dt))/(2*dt)
        assert abs(derivative-10*(.3-continuous_boundary(t)))<2e-9
    assert abs(25/.321-77.8816199376947)<1e-12
    for args in [([],[],[1]),([1],[1,0],[0]),([1.1],[1],[0])]:
        try:
            ppi_pooled(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid PPI input accepted")
    for args in [(0.,1.,0.),(.4,1.,.2),(0.,1.,float('nan'))]:
        try:
            held_velocity(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid held-control input accepted")
    print("PASS: exhaustive PPI, binomial bound, semidefinite stability, sampled CBF and robust held-input limit")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.check:
        check()
        return
    for n in (20,100,1000):
        print(f"zero failures / {n}: one-sided 95% failure upper bound = {zero_failure_upper(n):.6%}")
    print(f"CBF u(0)={scalar_cbf(0.,3.):.1f}; held for 0.2 s -> x=0.600000 m")
    print(f"continuously recomputed boundary control -> x(0.2)={continuous_boundary(.2):.6f} m")
    print(f"held-input limit for 0.2 s: u={held_velocity(0.,3.,.2):.1f} m/s; with d<=0.2 m/s: u={held_velocity(0.,3.,.2,disturbance_upper=.2):.1f} m/s")
    print(f"25 actions / 0.321 s = {25/.321:.3f} actions/s; queries/s = {1/.321:.3f}")


if __name__=='__main__':
    main()
