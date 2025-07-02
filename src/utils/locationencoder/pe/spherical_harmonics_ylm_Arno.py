"""
analytic expressions of spherical harmonics generated with sympy file 
Marc Russwurm generated 2024-10-25

run 
python spherical_harmonics_generate_ylms.py > spherical_harmonics_ylm.py

to generate the source code
"""

import torch
from torch import cos, sin

def get_SH(m,l):
  fname = f"Yl{l}_m{m}".replace("-","_minus_")
  return globals()[fname]

def SH(m, l, phi, theta):
  Ylm = get_SH(m,l)
  return Ylm(theta, phi)



@torch.jit.script
def Yl0_m0(theta, phi):
    return 0.282094791773878

@torch.jit.script
def Yl1_m_minus_1(theta, phi):
    return 0.48860251190292*(1.0 - cos(theta)**2)**0.5*sin(phi)

@torch.jit.script
def Yl1_m0(theta, phi):
    return 0.48860251190292*cos(theta)

@torch.jit.script
def Yl1_m1(theta, phi):
    return 0.48860251190292*(1.0 - cos(theta)**2)**0.5*cos(phi)

@torch.jit.script
def Yl2_m_minus_2(theta, phi):
    return 0.18209140509868*(3.0 - 3.0*cos(theta)**2)*sin(2*phi)

@torch.jit.script
def Yl2_m_minus_1(theta, phi):
    return 1.09254843059208*(1.0 - cos(theta)**2)**0.5*sin(phi)*cos(theta)

@torch.jit.script
def Yl2_m0(theta, phi):
    return 0.94617469575756*cos(theta)**2 - 0.31539156525252

@torch.jit.script
def Yl2_m1(theta, phi):
    return 1.09254843059208*(1.0 - cos(theta)**2)**0.5*cos(phi)*cos(theta)

@torch.jit.script
def Yl2_m2(theta, phi):
    return 0.18209140509868*(3.0 - 3.0*cos(theta)**2)*cos(2*phi)

@torch.jit.script
def Yl3_m_minus_3(theta, phi):
    return 0.590043589926644*(1.0 - cos(theta)**2)**1.5*sin(3*phi)

@torch.jit.script
def Yl3_m_minus_2(theta, phi):
    return 1.44530572132028*(1.0 - cos(theta)**2)*sin(2*phi)*cos(theta)

@torch.jit.script
def Yl3_m_minus_1(theta, phi):
    return 0.304697199642977*(1.0 - cos(theta)**2)**0.5*(7.5*cos(theta)**2 - 1.5)*sin(phi)

@torch.jit.script
def Yl3_m0(theta, phi):
    return 1.86588166295058*cos(theta)**3 - 1.11952899777035*cos(theta)

@torch.jit.script
def Yl3_m1(theta, phi):
    return 0.304697199642977*(1.0 - cos(theta)**2)**0.5*(7.5*cos(theta)**2 - 1.5)*cos(phi)

@torch.jit.script
def Yl3_m2(theta, phi):
    return 1.44530572132028*(1.0 - cos(theta)**2)*cos(2*phi)*cos(theta)

@torch.jit.script
def Yl3_m3(theta, phi):
    return 0.590043589926644*(1.0 - cos(theta)**2)**1.5*cos(3*phi)

@torch.jit.script
def Yl4_m_minus_4(theta, phi):
    return 0.625835735449176*(1.0 - cos(theta)**2)**2*sin(4*phi)

@torch.jit.script
def Yl4_m_minus_3(theta, phi):
    return 1.77013076977993*(1.0 - cos(theta)**2)**1.5*sin(3*phi)*cos(theta)

@torch.jit.script
def Yl4_m_minus_2(theta, phi):
    return 0.063078313050504*(1.0 - cos(theta)**2)*(52.5*cos(theta)**2 - 7.5)*sin(2*phi)

@torch.jit.script
def Yl4_m_minus_1(theta, phi):
    return 0.267618617422916*(1.0 - cos(theta)**2)**0.5*(17.5*cos(theta)**3 - 7.5*cos(theta))*sin(phi)

@torch.jit.script
def Yl4_m0(theta, phi):
    return 3.70249414203215*cos(theta)**4 - 3.17356640745613*cos(theta)**2 + 0.317356640745613

@torch.jit.script
def Yl4_m1(theta, phi):
    return 0.267618617422916*(1.0 - cos(theta)**2)**0.5*(17.5*cos(theta)**3 - 7.5*cos(theta))*cos(phi)

@torch.jit.script
def Yl4_m2(theta, phi):
    return 0.063078313050504*(1.0 - cos(theta)**2)*(52.5*cos(theta)**2 - 7.5)*cos(2*phi)

@torch.jit.script
def Yl4_m3(theta, phi):
    return 1.77013076977993*(1.0 - cos(theta)**2)**1.5*cos(3*phi)*cos(theta)

@torch.jit.script
def Yl4_m4(theta, phi):
    return 0.625835735449176*(1.0 - cos(theta)**2)**2*cos(4*phi)

@torch.jit.script
def Yl5_m_minus_5(theta, phi):
    return 0.65638205684017*(1.0 - cos(theta)**2)**2.5*sin(5*phi)

@torch.jit.script
def Yl5_m_minus_4(theta, phi):
    return 2.07566231488104*(1.0 - cos(theta)**2)**2*sin(4*phi)*cos(theta)

@torch.jit.script
def Yl5_m_minus_3(theta, phi):
    return 0.00931882475114763*(1.0 - cos(theta)**2)**1.5*(472.5*cos(theta)**2 - 52.5)*sin(3*phi)

@torch.jit.script
def Yl5_m_minus_2(theta, phi):
    return 0.0456527312854602*(1.0 - cos(theta)**2)*(157.5*cos(theta)**3 - 52.5*cos(theta))*sin(2*phi)

@torch.jit.script
def Yl5_m_minus_1(theta, phi):
    return 0.241571547304372*(1.0 - cos(theta)**2)**0.5*(39.375*cos(theta)**4 - 26.25*cos(theta)**2 + 1.875)*sin(phi)

@torch.jit.script
def Yl5_m0(theta, phi):
    return 7.36787031456569*cos(theta)**5 - 8.18652257173965*cos(theta)**3 + 1.75425483680135*cos(theta)

@torch.jit.script
def Yl5_m1(theta, phi):
    return 0.241571547304372*(1.0 - cos(theta)**2)**0.5*(39.375*cos(theta)**4 - 26.25*cos(theta)**2 + 1.875)*cos(phi)

@torch.jit.script
def Yl5_m2(theta, phi):
    return 0.0456527312854602*(1.0 - cos(theta)**2)*(157.5*cos(theta)**3 - 52.5*cos(theta))*cos(2*phi)

@torch.jit.script
def Yl5_m3(theta, phi):
    return 0.00931882475114763*(1.0 - cos(theta)**2)**1.5*(472.5*cos(theta)**2 - 52.5)*cos(3*phi)

@torch.jit.script
def Yl5_m4(theta, phi):
    return 2.07566231488104*(1.0 - cos(theta)**2)**2*cos(4*phi)*cos(theta)

@torch.jit.script
def Yl5_m5(theta, phi):
    return 0.65638205684017*(1.0 - cos(theta)**2)**2.5*cos(5*phi)

@torch.jit.script
def Yl6_m_minus_6(theta, phi):
    return 0.683184105191914*(1.0 - cos(theta)**2)**3*sin(6*phi)

@torch.jit.script
def Yl6_m_minus_5(theta, phi):
    return 2.36661916223175*(1.0 - cos(theta)**2)**2.5*sin(5*phi)*cos(theta)

@torch.jit.script
def Yl6_m_minus_4(theta, phi):
    return 0.0010678622237645*(1.0 - cos(theta)**2)**2*(5197.5*cos(theta)**2 - 472.5)*sin(4*phi)

@torch.jit.script
def Yl6_m_minus_3(theta, phi):
    return 0.00584892228263444*(1.0 - cos(theta)**2)**1.5*(1732.5*cos(theta)**3 - 472.5*cos(theta))*sin(3*phi)

@torch.jit.script
def Yl6_m_minus_2(theta, phi):
    return 0.0350935336958066*(1.0 - cos(theta)**2)*(433.125*cos(theta)**4 - 236.25*cos(theta)**2 + 13.125)*sin(2*phi)

@torch.jit.script
def Yl6_m_minus_1(theta, phi):
    return 0.221950995245231*(1.0 - cos(theta)**2)**0.5*(86.625*cos(theta)**5 - 78.75*cos(theta)**3 + 13.125*cos(theta))*sin(phi)

@torch.jit.script
def Yl6_m0(theta, phi):
    return 14.6844857238222*cos(theta)**6 - 20.024298714303*cos(theta)**4 + 6.67476623810098*cos(theta)**2 - 0.317846011338142

@torch.jit.script
def Yl6_m1(theta, phi):
    return 0.221950995245231*(1.0 - cos(theta)**2)**0.5*(86.625*cos(theta)**5 - 78.75*cos(theta)**3 + 13.125*cos(theta))*cos(phi)

@torch.jit.script
def Yl6_m2(theta, phi):
    return 0.0350935336958066*(1.0 - cos(theta)**2)*(433.125*cos(theta)**4 - 236.25*cos(theta)**2 + 13.125)*cos(2*phi)

@torch.jit.script
def Yl6_m3(theta, phi):
    return 0.00584892228263444*(1.0 - cos(theta)**2)**1.5*(1732.5*cos(theta)**3 - 472.5*cos(theta))*cos(3*phi)

@torch.jit.script
def Yl6_m4(theta, phi):
    return 0.0010678622237645*(1.0 - cos(theta)**2)**2*(5197.5*cos(theta)**2 - 472.5)*cos(4*phi)

@torch.jit.script
def Yl6_m5(theta, phi):
    return 2.36661916223175*(1.0 - cos(theta)**2)**2.5*cos(5*phi)*cos(theta)

@torch.jit.script
def Yl6_m6(theta, phi):
    return 0.683184105191914*(1.0 - cos(theta)**2)**3*cos(6*phi)

@torch.jit.script
def Yl7_m_minus_7(theta, phi):
    return 0.707162732524596*(1.0 - cos(theta)**2)**3.5*sin(7*phi)

@torch.jit.script
def Yl7_m_minus_6(theta, phi):
    return 2.6459606618019*(1.0 - cos(theta)**2)**3*sin(6*phi)*cos(theta)

@torch.jit.script
def Yl7_m_minus_5(theta, phi):
    return 9.98394571852353e-5*(1.0 - cos(theta)**2)**2.5*(67567.5*cos(theta)**2 - 5197.5)*sin(5*phi)

@torch.jit.script
def Yl7_m_minus_4(theta, phi):
    return 0.000599036743111412*(1.0 - cos(theta)**2)**2*(22522.5*cos(theta)**3 - 5197.5*cos(theta))*sin(4*phi)

@torch.jit.script
def Yl7_m_minus_3(theta, phi):
    return 0.00397356022507413*(1.0 - cos(theta)**2)**1.5*(5630.625*cos(theta)**4 - 2598.75*cos(theta)**2 + 118.125)*sin(3*phi)

@torch.jit.script
def Yl7_m_minus_2(theta, phi):
    return 0.0280973138060306*(1.0 - cos(theta)**2)*(1126.125*cos(theta)**5 - 866.25*cos(theta)**3 + 118.125*cos(theta))*sin(2*phi)

@torch.jit.script
def Yl7_m_minus_1(theta, phi):
    return 0.206472245902897*(1.0 - cos(theta)**2)**0.5*(187.6875*cos(theta)**6 - 216.5625*cos(theta)**4 + 59.0625*cos(theta)**2 - 2.1875)*sin(phi)

@torch.jit.script
def Yl7_m0(theta, phi):
    return 29.2939547952501*cos(theta)**7 - 47.3210039000194*cos(theta)**5 + 21.5095472272816*cos(theta)**3 - 2.38994969192017*cos(theta)

@torch.jit.script
def Yl7_m1(theta, phi):
    return 0.206472245902897*(1.0 - cos(theta)**2)**0.5*(187.6875*cos(theta)**6 - 216.5625*cos(theta)**4 + 59.0625*cos(theta)**2 - 2.1875)*cos(phi)

@torch.jit.script
def Yl7_m2(theta, phi):
    return 0.0280973138060306*(1.0 - cos(theta)**2)*(1126.125*cos(theta)**5 - 866.25*cos(theta)**3 + 118.125*cos(theta))*cos(2*phi)

@torch.jit.script
def Yl7_m3(theta, phi):
    return 0.00397356022507413*(1.0 - cos(theta)**2)**1.5*(5630.625*cos(theta)**4 - 2598.75*cos(theta)**2 + 118.125)*cos(3*phi)

@torch.jit.script
def Yl7_m4(theta, phi):
    return 0.000599036743111412*(1.0 - cos(theta)**2)**2*(22522.5*cos(theta)**3 - 5197.5*cos(theta))*cos(4*phi)

@torch.jit.script
def Yl7_m5(theta, phi):
    return 9.98394571852353e-5*(1.0 - cos(theta)**2)**2.5*(67567.5*cos(theta)**2 - 5197.5)*cos(5*phi)

@torch.jit.script
def Yl7_m6(theta, phi):
    return 2.6459606618019*(1.0 - cos(theta)**2)**3*cos(6*phi)*cos(theta)

@torch.jit.script
def Yl7_m7(theta, phi):
    return 0.707162732524596*(1.0 - cos(theta)**2)**3.5*cos(7*phi)

@torch.jit.script
def Yl8_m_minus_8(theta, phi):
    return 0.72892666017483*(1.0 - cos(theta)**2)**4*sin(8*phi)

@torch.jit.script
def Yl8_m_minus_7(theta, phi):
    return 2.91570664069932*(1.0 - cos(theta)**2)**3.5*sin(7*phi)*cos(theta)

@torch.jit.script
def Yl8_m_minus_6(theta, phi):
    return 7.87853281621404e-6*(1.0 - cos(theta)**2)**3*(1013512.5*cos(theta)**2 - 67567.5)*sin(6*phi)

@torch.jit.script
def Yl8_m_minus_5(theta, phi):
    return 5.10587282657803e-5*(1.0 - cos(theta)**2)**2.5*(337837.5*cos(theta)**3 - 67567.5*cos(theta))*sin(5*phi)

@torch.jit.script
def Yl8_m_minus_4(theta, phi):
    return 0.000368189725644507*(1.0 - cos(theta)**2)**2*(84459.375*cos(theta)**4 - 33783.75*cos(theta)**2 + 1299.375)*sin(4*phi)

@torch.jit.script
def Yl8_m_minus_3(theta, phi):
    return 0.0028519853513317*(1.0 - cos(theta)**2)**1.5*(16891.875*cos(theta)**5 - 11261.25*cos(theta)**3 + 1299.375*cos(theta))*sin(3*phi)

@torch.jit.script
def Yl8_m_minus_2(theta, phi):
    return 0.0231696385236779*(1.0 - cos(theta)**2)*(2815.3125*cos(theta)**6 - 2815.3125*cos(theta)**4 + 649.6875*cos(theta)**2 - 19.6875)*sin(2*phi)

@torch.jit.script
def Yl8_m_minus_1(theta, phi):
    return 0.193851103820053*(1.0 - cos(theta)**2)**0.5*(402.1875*cos(theta)**7 - 563.0625*cos(theta)**5 + 216.5625*cos(theta)**3 - 19.6875*cos(theta))*sin(phi)

@torch.jit.script
def Yl8_m0(theta, phi):
    return 58.4733681132208*cos(theta)**8 - 109.150287144679*cos(theta)**6 + 62.9713195065454*cos(theta)**4 - 11.4493308193719*cos(theta)**2 + 0.318036967204775

@torch.jit.script
def Yl8_m1(theta, phi):
    return 0.193851103820053*(1.0 - cos(theta)**2)**0.5*(402.1875*cos(theta)**7 - 563.0625*cos(theta)**5 + 216.5625*cos(theta)**3 - 19.6875*cos(theta))*cos(phi)

@torch.jit.script
def Yl8_m2(theta, phi):
    return 0.0231696385236779*(1.0 - cos(theta)**2)*(2815.3125*cos(theta)**6 - 2815.3125*cos(theta)**4 + 649.6875*cos(theta)**2 - 19.6875)*cos(2*phi)

@torch.jit.script
def Yl8_m3(theta, phi):
    return 0.0028519853513317*(1.0 - cos(theta)**2)**1.5*(16891.875*cos(theta)**5 - 11261.25*cos(theta)**3 + 1299.375*cos(theta))*cos(3*phi)

@torch.jit.script
def Yl8_m4(theta, phi):
    return 0.000368189725644507*(1.0 - cos(theta)**2)**2*(84459.375*cos(theta)**4 - 33783.75*cos(theta)**2 + 1299.375)*cos(4*phi)

@torch.jit.script
def Yl8_m5(theta, phi):
    return 5.10587282657803e-5*(1.0 - cos(theta)**2)**2.5*(337837.5*cos(theta)**3 - 67567.5*cos(theta))*cos(5*phi)

@torch.jit.script
def Yl8_m6(theta, phi):
    return 7.87853281621404e-6*(1.0 - cos(theta)**2)**3*(1013512.5*cos(theta)**2 - 67567.5)*cos(6*phi)

@torch.jit.script
def Yl8_m7(theta, phi):
    return 2.91570664069932*(1.0 - cos(theta)**2)**3.5*cos(7*phi)*cos(theta)

@torch.jit.script
def Yl8_m8(theta, phi):
    return 0.72892666017483*(1.0 - cos(theta)**2)**4*cos(8*phi)

@torch.jit.script
def Yl9_m_minus_9(theta, phi):
    return 0.748900951853188*(1.0 - cos(theta)**2)**4.5*sin(9*phi)

@torch.jit.script
def Yl9_m_minus_8(theta, phi):
    return 3.1773176489547*(1.0 - cos(theta)**2)**4*sin(8*phi)*cos(theta)

@torch.jit.script
def Yl9_m_minus_7(theta, phi):
    return 5.37640612566745e-7*(1.0 - cos(theta)**2)**3.5*(17229712.5*cos(theta)**2 - 1013512.5)*sin(7*phi)

@torch.jit.script
def Yl9_m_minus_6(theta, phi):
    return 3.72488342871223e-6*(1.0 - cos(theta)**2)**3*(5743237.5*cos(theta)**3 - 1013512.5*cos(theta))*sin(6*phi)

@torch.jit.script
def Yl9_m_minus_5(theta, phi):
    return 2.88528229719329e-5*(1.0 - cos(theta)**2)**2.5*(1435809.375*cos(theta)**4 - 506756.25*cos(theta)**2 + 16891.875)*sin(5*phi)

@torch.jit.script
def Yl9_m_minus_4(theta, phi):
    return 0.000241400036332803*(1.0 - cos(theta)**2)**2*(287161.875*cos(theta)**5 - 168918.75*cos(theta)**3 + 16891.875*cos(theta))*sin(4*phi)

@torch.jit.script
def Yl9_m_minus_3(theta, phi):
    return 0.00213198739401417*(1.0 - cos(theta)**2)**1.5*(47860.3125*cos(theta)**6 - 42229.6875*cos(theta)**4 + 8445.9375*cos(theta)**2 - 216.5625)*sin(3*phi)

@torch.jit.script
def Yl9_m_minus_2(theta, phi):
    return 0.0195399872275232*(1.0 - cos(theta)**2)*(6837.1875*cos(theta)**7 - 8445.9375*cos(theta)**5 + 2815.3125*cos(theta)**3 - 216.5625*cos(theta))*sin(2*phi)

@torch.jit.script
def Yl9_m_minus_1(theta, phi):
    return 0.183301328077446*(1.0 - cos(theta)**2)**0.5*(854.6484375*cos(theta)**8 - 1407.65625*cos(theta)**6 + 703.828125*cos(theta)**4 - 108.28125*cos(theta)**2 + 2.4609375)*sin(phi)

@torch.jit.script
def Yl9_m0(theta, phi):
    return 116.766123398619*cos(theta)**9 - 247.269437785311*cos(theta)**7 + 173.088606449718*cos(theta)**5 - 44.3816939614661*cos(theta)**3 + 3.02602458828178*cos(theta)

@torch.jit.script
def Yl9_m1(theta, phi):
    return 0.183301328077446*(1.0 - cos(theta)**2)**0.5*(854.6484375*cos(theta)**8 - 1407.65625*cos(theta)**6 + 703.828125*cos(theta)**4 - 108.28125*cos(theta)**2 + 2.4609375)*cos(phi)

@torch.jit.script
def Yl9_m2(theta, phi):
    return 0.0195399872275232*(1.0 - cos(theta)**2)*(6837.1875*cos(theta)**7 - 8445.9375*cos(theta)**5 + 2815.3125*cos(theta)**3 - 216.5625*cos(theta))*cos(2*phi)

@torch.jit.script
def Yl9_m3(theta, phi):
    return 0.00213198739401417*(1.0 - cos(theta)**2)**1.5*(47860.3125*cos(theta)**6 - 42229.6875*cos(theta)**4 + 8445.9375*cos(theta)**2 - 216.5625)*cos(3*phi)

@torch.jit.script
def Yl9_m4(theta, phi):
    return 0.000241400036332803*(1.0 - cos(theta)**2)**2*(287161.875*cos(theta)**5 - 168918.75*cos(theta)**3 + 16891.875*cos(theta))*cos(4*phi)

@torch.jit.script
def Yl9_m5(theta, phi):
    return 2.88528229719329e-5*(1.0 - cos(theta)**2)**2.5*(1435809.375*cos(theta)**4 - 506756.25*cos(theta)**2 + 16891.875)*cos(5*phi)

@torch.jit.script
def Yl9_m6(theta, phi):
    return 3.72488342871223e-6*(1.0 - cos(theta)**2)**3*(5743237.5*cos(theta)**3 - 1013512.5*cos(theta))*cos(6*phi)

@torch.jit.script
def Yl9_m7(theta, phi):
    return 5.37640612566745e-7*(1.0 - cos(theta)**2)**3.5*(17229712.5*cos(theta)**2 - 1013512.5)*cos(7*phi)

@torch.jit.script
def Yl9_m8(theta, phi):
    return 3.1773176489547*(1.0 - cos(theta)**2)**4*cos(8*phi)*cos(theta)

@torch.jit.script
def Yl9_m9(theta, phi):
    return 0.748900951853188*(1.0 - cos(theta)**2)**4.5*cos(9*phi)

@torch.jit.script
def Yl10_m_minus_10(theta, phi):
    return 0.76739511822199*(1.0 - cos(theta)**2)**5*sin(10*phi)

@torch.jit.script
def Yl10_m_minus_9(theta, phi):
    return 3.43189529989171*(1.0 - cos(theta)**2)**4.5*sin(9*phi)*cos(theta)

@torch.jit.script
def Yl10_m_minus_8(theta, phi):
    return 3.23120268385452e-8*(1.0 - cos(theta)**2)**4*(327364537.5*cos(theta)**2 - 17229712.5)*sin(8*phi)

@torch.jit.script
def Yl10_m_minus_7(theta, phi):
    return 2.37443934928654e-7*(1.0 - cos(theta)**2)**3.5*(109121512.5*cos(theta)**3 - 17229712.5*cos(theta))*sin(7*phi)

@torch.jit.script
def Yl10_m_minus_6(theta, phi):
    return 1.95801284774625e-6*(1.0 - cos(theta)**2)**3*(27280378.125*cos(theta)**4 - 8614856.25*cos(theta)**2 + 253378.125)*sin(6*phi)

@torch.jit.script
def Yl10_m_minus_5(theta, phi):
    return 1.75129993135143e-5*(1.0 - cos(theta)**2)**2.5*(5456075.625*cos(theta)**5 - 2871618.75*cos(theta)**3 + 253378.125*cos(theta))*sin(5*phi)

@torch.jit.script
def Yl10_m_minus_4(theta, phi):
    return 0.000166142899475011*(1.0 - cos(theta)**2)**2*(909345.9375*cos(theta)**6 - 717904.6875*cos(theta)**4 + 126689.0625*cos(theta)**2 - 2815.3125)*sin(4*phi)

@torch.jit.script
def Yl10_m_minus_3(theta, phi):
    return 0.00164473079210685*(1.0 - cos(theta)**2)**1.5*(129906.5625*cos(theta)**7 - 143580.9375*cos(theta)**5 + 42229.6875*cos(theta)**3 - 2815.3125*cos(theta))*sin(3*phi)

@torch.jit.script
def Yl10_m_minus_2(theta, phi):
    return 0.0167730288071195*(1.0 - cos(theta)**2)*(16238.3203125*cos(theta)**8 - 23930.15625*cos(theta)**6 + 10557.421875*cos(theta)**4 - 1407.65625*cos(theta)**2 + 27.0703125)*sin(2*phi)

@torch.jit.script
def Yl10_m_minus_1(theta, phi):
    return 0.174310428544485*(1.0 - cos(theta)**2)**0.5*(1804.2578125*cos(theta)**9 - 3418.59375*cos(theta)**7 + 2111.484375*cos(theta)**5 - 469.21875*cos(theta)**3 + 27.0703125*cos(theta))*sin(phi)

@torch.jit.script
def Yl10_m0(theta, phi):
    return 233.240148813258*cos(theta)**10 - 552.410878768242*cos(theta)**8 + 454.926606044435*cos(theta)**6 - 151.642202014812*cos(theta)**4 + 17.4971771555552*cos(theta)**2 - 0.318130493737367

@torch.jit.script
def Yl10_m1(theta, phi):
    return 0.174310428544485*(1.0 - cos(theta)**2)**0.5*(1804.2578125*cos(theta)**9 - 3418.59375*cos(theta)**7 + 2111.484375*cos(theta)**5 - 469.21875*cos(theta)**3 + 27.0703125*cos(theta))*cos(phi)

@torch.jit.script
def Yl10_m2(theta, phi):
    return 0.0167730288071195*(1.0 - cos(theta)**2)*(16238.3203125*cos(theta)**8 - 23930.15625*cos(theta)**6 + 10557.421875*cos(theta)**4 - 1407.65625*cos(theta)**2 + 27.0703125)*cos(2*phi)

@torch.jit.script
def Yl10_m3(theta, phi):
    return 0.00164473079210685*(1.0 - cos(theta)**2)**1.5*(129906.5625*cos(theta)**7 - 143580.9375*cos(theta)**5 + 42229.6875*cos(theta)**3 - 2815.3125*cos(theta))*cos(3*phi)

@torch.jit.script
def Yl10_m4(theta, phi):
    return 0.000166142899475011*(1.0 - cos(theta)**2)**2*(909345.9375*cos(theta)**6 - 717904.6875*cos(theta)**4 + 126689.0625*cos(theta)**2 - 2815.3125)*cos(4*phi)

@torch.jit.script
def Yl10_m5(theta, phi):
    return 1.75129993135143e-5*(1.0 - cos(theta)**2)**2.5*(5456075.625*cos(theta)**5 - 2871618.75*cos(theta)**3 + 253378.125*cos(theta))*cos(5*phi)

@torch.jit.script
def Yl10_m6(theta, phi):
    return 1.95801284774625e-6*(1.0 - cos(theta)**2)**3*(27280378.125*cos(theta)**4 - 8614856.25*cos(theta)**2 + 253378.125)*cos(6*phi)

@torch.jit.script
def Yl10_m7(theta, phi):
    return 2.37443934928654e-7*(1.0 - cos(theta)**2)**3.5*(109121512.5*cos(theta)**3 - 17229712.5*cos(theta))*cos(7*phi)

@torch.jit.script
def Yl10_m8(theta, phi):
    return 3.23120268385452e-8*(1.0 - cos(theta)**2)**4*(327364537.5*cos(theta)**2 - 17229712.5)*cos(8*phi)

@torch.jit.script
def Yl10_m9(theta, phi):
    return 3.43189529989171*(1.0 - cos(theta)**2)**4.5*cos(9*phi)*cos(theta)

@torch.jit.script
def Yl10_m10(theta, phi):
    return 0.76739511822199*(1.0 - cos(theta)**2)**5*cos(10*phi)

@torch.jit.script
def Yl11_m_minus_11(theta, phi):
    return 0.784642105787197*(1.0 - cos(theta)**2)**5.5*sin(11*phi)

@torch.jit.script
def Yl11_m_minus_10(theta, phi):
    return 3.68029769880531*(1.0 - cos(theta)**2)**5*sin(10*phi)*cos(theta)

@torch.jit.script
def Yl11_m_minus_9(theta, phi):
    return 1.73470916587426e-9*(1.0 - cos(theta)**2)**4.5*(6874655287.5*cos(theta)**2 - 327364537.5)*sin(9*phi)

@torch.jit.script
def Yl11_m_minus_8(theta, phi):
    return 1.34369994198887e-8*(1.0 - cos(theta)**2)**4*(2291551762.5*cos(theta)**3 - 327364537.5*cos(theta))*sin(8*phi)

@torch.jit.script
def Yl11_m_minus_7(theta, phi):
    return 1.17141045151419e-7*(1.0 - cos(theta)**2)**3.5*(572887940.625*cos(theta)**4 - 163682268.75*cos(theta)**2 + 4307428.125)*sin(7*phi)

@torch.jit.script
def Yl11_m_minus_6(theta, phi):
    return 1.11129753051333e-6*(1.0 - cos(theta)**2)**3*(114577588.125*cos(theta)**5 - 54560756.25*cos(theta)**3 + 4307428.125*cos(theta))*sin(6*phi)

@torch.jit.script
def Yl11_m_minus_5(theta, phi):
    return 1.12235548974089e-5*(1.0 - cos(theta)**2)**2.5*(19096264.6875*cos(theta)**6 - 13640189.0625*cos(theta)**4 + 2153714.0625*cos(theta)**2 - 42229.6875)*sin(5*phi)

@torch.jit.script
def Yl11_m_minus_4(theta, phi):
    return 0.0001187789403385*(1.0 - cos(theta)**2)**2*(2728037.8125*cos(theta)**7 - 2728037.8125*cos(theta)**5 + 717904.6875*cos(theta)**3 - 42229.6875*cos(theta))*sin(4*phi)

@torch.jit.script
def Yl11_m_minus_3(theta, phi):
    return 0.00130115809959914*(1.0 - cos(theta)**2)**1.5*(341004.7265625*cos(theta)**8 - 454672.96875*cos(theta)**6 + 179476.171875*cos(theta)**4 - 21114.84375*cos(theta)**2 + 351.9140625)*sin(3*phi)

@torch.jit.script
def Yl11_m_minus_2(theta, phi):
    return 0.0146054634441776*(1.0 - cos(theta)**2)*(37889.4140625*cos(theta)**9 - 64953.28125*cos(theta)**7 + 35895.234375*cos(theta)**5 - 7038.28125*cos(theta)**3 + 351.9140625*cos(theta))*sin(2*phi)

@torch.jit.script
def Yl11_m_minus_1(theta, phi):
    return 0.166527904912351*(1.0 - cos(theta)**2)**0.5*(3788.94140625*cos(theta)**10 - 8119.16015625*cos(theta)**8 + 5982.5390625*cos(theta)**6 - 1759.5703125*cos(theta)**4 + 175.95703125*cos(theta)**2 - 2.70703125)*sin(phi)

@torch.jit.script
def Yl11_m0(theta, phi):
    return 465.998147319252*cos(theta)**11 - 1220.47133821709*cos(theta)**9 + 1156.23600462672*cos(theta)**7 - 476.097178375706*cos(theta)**5 + 79.3495297292844*cos(theta)**3 - 3.66228598750543*cos(theta)

@torch.jit.script
def Yl11_m1(theta, phi):
    return 0.166527904912351*(1.0 - cos(theta)**2)**0.5*(3788.94140625*cos(theta)**10 - 8119.16015625*cos(theta)**8 + 5982.5390625*cos(theta)**6 - 1759.5703125*cos(theta)**4 + 175.95703125*cos(theta)**2 - 2.70703125)*cos(phi)

@torch.jit.script
def Yl11_m2(theta, phi):
    return 0.0146054634441776*(1.0 - cos(theta)**2)*(37889.4140625*cos(theta)**9 - 64953.28125*cos(theta)**7 + 35895.234375*cos(theta)**5 - 7038.28125*cos(theta)**3 + 351.9140625*cos(theta))*cos(2*phi)

@torch.jit.script
def Yl11_m3(theta, phi):
    return 0.00130115809959914*(1.0 - cos(theta)**2)**1.5*(341004.7265625*cos(theta)**8 - 454672.96875*cos(theta)**6 + 179476.171875*cos(theta)**4 - 21114.84375*cos(theta)**2 + 351.9140625)*cos(3*phi)

@torch.jit.script
def Yl11_m4(theta, phi):
    return 0.0001187789403385*(1.0 - cos(theta)**2)**2*(2728037.8125*cos(theta)**7 - 2728037.8125*cos(theta)**5 + 717904.6875*cos(theta)**3 - 42229.6875*cos(theta))*cos(4*phi)

@torch.jit.script
def Yl11_m5(theta, phi):
    return 1.12235548974089e-5*(1.0 - cos(theta)**2)**2.5*(19096264.6875*cos(theta)**6 - 13640189.0625*cos(theta)**4 + 2153714.0625*cos(theta)**2 - 42229.6875)*cos(5*phi)

@torch.jit.script
def Yl11_m6(theta, phi):
    return 1.11129753051333e-6*(1.0 - cos(theta)**2)**3*(114577588.125*cos(theta)**5 - 54560756.25*cos(theta)**3 + 4307428.125*cos(theta))*cos(6*phi)

@torch.jit.script
def Yl11_m7(theta, phi):
    return 1.17141045151419e-7*(1.0 - cos(theta)**2)**3.5*(572887940.625*cos(theta)**4 - 163682268.75*cos(theta)**2 + 4307428.125)*cos(7*phi)

@torch.jit.script
def Yl11_m8(theta, phi):
    return 1.34369994198887e-8*(1.0 - cos(theta)**2)**4*(2291551762.5*cos(theta)**3 - 327364537.5*cos(theta))*cos(8*phi)

@torch.jit.script
def Yl11_m9(theta, phi):
    return 1.73470916587426e-9*(1.0 - cos(theta)**2)**4.5*(6874655287.5*cos(theta)**2 - 327364537.5)*cos(9*phi)

@torch.jit.script
def Yl11_m10(theta, phi):
    return 3.68029769880531*(1.0 - cos(theta)**2)**5*cos(10*phi)*cos(theta)

@torch.jit.script
def Yl11_m11(theta, phi):
    return 0.784642105787197*(1.0 - cos(theta)**2)**5.5*cos(11*phi)

@torch.jit.script
def Yl12_m_minus_12(theta, phi):
    return 0.800821995783972*(1.0 - cos(theta)**2)**6*sin(12*phi)

@torch.jit.script
def Yl12_m_minus_11(theta, phi):
    return 3.92321052893598*(1.0 - cos(theta)**2)**5.5*sin(11*phi)*cos(theta)

@torch.jit.script
def Yl12_m_minus_10(theta, phi):
    return 8.4141794839602e-11*(1.0 - cos(theta)**2)**5*(158117071612.5*cos(theta)**2 - 6874655287.5)*sin(10*phi)

@torch.jit.script
def Yl12_m_minus_9(theta, phi):
    return 6.83571172711927e-10*(1.0 - cos(theta)**2)**4.5*(52705690537.5*cos(theta)**3 - 6874655287.5*cos(theta))*sin(9*phi)

@torch.jit.script
def Yl12_m_minus_8(theta, phi):
    return 6.26503328368427e-9*(1.0 - cos(theta)**2)**4*(13176422634.375*cos(theta)**4 - 3437327643.75*cos(theta)**2 + 81841134.375)*sin(8*phi)

@torch.jit.script
def Yl12_m_minus_7(theta, phi):
    return 6.26503328368427e-8*(1.0 - cos(theta)**2)**3.5*(2635284526.875*cos(theta)**5 - 1145775881.25*cos(theta)**3 + 81841134.375*cos(theta))*sin(7*phi)

@torch.jit.script
def Yl12_m_minus_6(theta, phi):
    return 6.68922506214776e-7*(1.0 - cos(theta)**2)**3*(439214087.8125*cos(theta)**6 - 286443970.3125*cos(theta)**4 + 40920567.1875*cos(theta)**2 - 717904.6875)*sin(6*phi)

@torch.jit.script
def Yl12_m_minus_5(theta, phi):
    return 7.50863650967357e-6*(1.0 - cos(theta)**2)**2.5*(62744869.6875*cos(theta)**7 - 57288794.0625*cos(theta)**5 + 13640189.0625*cos(theta)**3 - 717904.6875*cos(theta))*sin(5*phi)

@torch.jit.script
def Yl12_m_minus_4(theta, phi):
    return 8.75649965675714e-5*(1.0 - cos(theta)**2)**2*(7843108.7109375*cos(theta)**8 - 9548132.34375*cos(theta)**6 + 3410047.265625*cos(theta)**4 - 358952.34375*cos(theta)**2 + 5278.7109375)*sin(4*phi)

@torch.jit.script
def Yl12_m_minus_3(theta, phi):
    return 0.00105077995881086*(1.0 - cos(theta)**2)**1.5*(871456.5234375*cos(theta)**9 - 1364018.90625*cos(theta)**7 + 682009.453125*cos(theta)**5 - 119650.78125*cos(theta)**3 + 5278.7109375*cos(theta))*sin(3*phi)

@torch.jit.script
def Yl12_m_minus_2(theta, phi):
    return 0.0128693736551466*(1.0 - cos(theta)**2)*(87145.65234375*cos(theta)**10 - 170502.36328125*cos(theta)**8 + 113668.2421875*cos(theta)**6 - 29912.6953125*cos(theta)**4 + 2639.35546875*cos(theta)**2 - 35.19140625)*sin(2*phi)

@torch.jit.script
def Yl12_m_minus_1(theta, phi):
    return 0.159704727088682*(1.0 - cos(theta)**2)**0.5*(7922.33203125*cos(theta)**11 - 18944.70703125*cos(theta)**9 + 16238.3203125*cos(theta)**7 - 5982.5390625*cos(theta)**5 + 879.78515625*cos(theta)**3 - 35.19140625*cos(theta))*sin(phi)

@torch.jit.script
def Yl12_m0(theta, phi):
    return 931.186918632914*cos(theta)**12 - 2672.1015925988*cos(theta)**10 + 2862.96599207014*cos(theta)**8 - 1406.36925926252*cos(theta)**6 + 310.228513072616*cos(theta)**4 - 24.8182810458093*cos(theta)**2 + 0.318183090330888

@torch.jit.script
def Yl12_m1(theta, phi):
    return 0.159704727088682*(1.0 - cos(theta)**2)**0.5*(7922.33203125*cos(theta)**11 - 18944.70703125*cos(theta)**9 + 16238.3203125*cos(theta)**7 - 5982.5390625*cos(theta)**5 + 879.78515625*cos(theta)**3 - 35.19140625*cos(theta))*cos(phi)

@torch.jit.script
def Yl12_m2(theta, phi):
    return 0.0128693736551466*(1.0 - cos(theta)**2)*(87145.65234375*cos(theta)**10 - 170502.36328125*cos(theta)**8 + 113668.2421875*cos(theta)**6 - 29912.6953125*cos(theta)**4 + 2639.35546875*cos(theta)**2 - 35.19140625)*cos(2*phi)

@torch.jit.script
def Yl12_m3(theta, phi):
    return 0.00105077995881086*(1.0 - cos(theta)**2)**1.5*(871456.5234375*cos(theta)**9 - 1364018.90625*cos(theta)**7 + 682009.453125*cos(theta)**5 - 119650.78125*cos(theta)**3 + 5278.7109375*cos(theta))*cos(3*phi)

@torch.jit.script
def Yl12_m4(theta, phi):
    return 8.75649965675714e-5*(1.0 - cos(theta)**2)**2*(7843108.7109375*cos(theta)**8 - 9548132.34375*cos(theta)**6 + 3410047.265625*cos(theta)**4 - 358952.34375*cos(theta)**2 + 5278.7109375)*cos(4*phi)

@torch.jit.script
def Yl12_m5(theta, phi):
    return 7.50863650967357e-6*(1.0 - cos(theta)**2)**2.5*(62744869.6875*cos(theta)**7 - 57288794.0625*cos(theta)**5 + 13640189.0625*cos(theta)**3 - 717904.6875*cos(theta))*cos(5*phi)

@torch.jit.script
def Yl12_m6(theta, phi):
    return 6.68922506214776e-7*(1.0 - cos(theta)**2)**3*(439214087.8125*cos(theta)**6 - 286443970.3125*cos(theta)**4 + 40920567.1875*cos(theta)**2 - 717904.6875)*cos(6*phi)

@torch.jit.script
def Yl12_m7(theta, phi):
    return 6.26503328368427e-8*(1.0 - cos(theta)**2)**3.5*(2635284526.875*cos(theta)**5 - 1145775881.25*cos(theta)**3 + 81841134.375*cos(theta))*cos(7*phi)

@torch.jit.script
def Yl12_m8(theta, phi):
    return 6.26503328368427e-9*(1.0 - cos(theta)**2)**4*(13176422634.375*cos(theta)**4 - 3437327643.75*cos(theta)**2 + 81841134.375)*cos(8*phi)

@torch.jit.script
def Yl12_m9(theta, phi):
    return 6.83571172711927e-10*(1.0 - cos(theta)**2)**4.5*(52705690537.5*cos(theta)**3 - 6874655287.5*cos(theta))*cos(9*phi)

@torch.jit.script
def Yl12_m10(theta, phi):
    return 8.4141794839602e-11*(1.0 - cos(theta)**2)**5*(158117071612.5*cos(theta)**2 - 6874655287.5)*cos(10*phi)

@torch.jit.script
def Yl12_m11(theta, phi):
    return 3.92321052893598*(1.0 - cos(theta)**2)**5.5*cos(11*phi)*cos(theta)

@torch.jit.script
def Yl12_m12(theta, phi):
    return 0.800821995783972*(1.0 - cos(theta)**2)**6*cos(12*phi)

@torch.jit.script
def Yl13_m_minus_13(theta, phi):
    return 0.816077118837628*(1.0 - cos(theta)**2)**6.5*sin(13*phi)

@torch.jit.script
def Yl13_m_minus_12(theta, phi):
    return 4.16119315354964*(1.0 - cos(theta)**2)**6*sin(12*phi)*cos(theta)

@torch.jit.script
def Yl13_m_minus_11(theta, phi):
    return 3.72180924766049e-12*(1.0 - cos(theta)**2)**5.5*(3952926790312.5*cos(theta)**2 - 158117071612.5)*sin(11*phi)

@torch.jit.script
def Yl13_m_minus_10(theta, phi):
    return 3.15805986876424e-11*(1.0 - cos(theta)**2)**5*(1317642263437.5*cos(theta)**3 - 158117071612.5*cos(theta))*sin(10*phi)

@torch.jit.script
def Yl13_m_minus_9(theta, phi):
    return 3.02910461422567e-10*(1.0 - cos(theta)**2)**4.5*(329410565859.375*cos(theta)**4 - 79058535806.25*cos(theta)**2 + 1718663821.875)*sin(9*phi)

@torch.jit.script
def Yl13_m_minus_8(theta, phi):
    return 3.17695172143292e-9*(1.0 - cos(theta)**2)**4*(65882113171.875*cos(theta)**5 - 26352845268.75*cos(theta)**3 + 1718663821.875*cos(theta))*sin(8*phi)

@torch.jit.script
def Yl13_m_minus_7(theta, phi):
    return 3.5661194627771e-8*(1.0 - cos(theta)**2)**3.5*(10980352195.3125*cos(theta)**6 - 6588211317.1875*cos(theta)**4 + 859331910.9375*cos(theta)**2 - 13640189.0625)*sin(7*phi)

@torch.jit.script
def Yl13_m_minus_6(theta, phi):
    return 4.21948945157073e-7*(1.0 - cos(theta)**2)**3*(1568621742.1875*cos(theta)**7 - 1317642263.4375*cos(theta)**5 + 286443970.3125*cos(theta)**3 - 13640189.0625*cos(theta))*sin(6*phi)

@torch.jit.script
def Yl13_m_minus_5(theta, phi):
    return 5.2021359721285e-6*(1.0 - cos(theta)**2)**2.5*(196077717.773438*cos(theta)**8 - 219607043.90625*cos(theta)**6 + 71610992.578125*cos(theta)**4 - 6820094.53125*cos(theta)**2 + 89738.0859375)*sin(5*phi)

@torch.jit.script
def Yl13_m_minus_4(theta, phi):
    return 6.62123812058377e-5*(1.0 - cos(theta)**2)**2*(21786413.0859375*cos(theta)**9 - 31372434.84375*cos(theta)**7 + 14322198.515625*cos(theta)**5 - 2273364.84375*cos(theta)**3 + 89738.0859375*cos(theta))*sin(4*phi)

@torch.jit.script
def Yl13_m_minus_3(theta, phi):
    return 0.000863303829622583*(1.0 - cos(theta)**2)**1.5*(2178641.30859375*cos(theta)**10 - 3921554.35546875*cos(theta)**8 + 2387033.0859375*cos(theta)**6 - 568341.2109375*cos(theta)**4 + 44869.04296875*cos(theta)**2 - 527.87109375)*sin(3*phi)

@torch.jit.script
def Yl13_m_minus_2(theta, phi):
    return 0.0114530195317401*(1.0 - cos(theta)**2)*(198058.30078125*cos(theta)**11 - 435728.26171875*cos(theta)**9 + 341004.7265625*cos(theta)**7 - 113668.2421875*cos(theta)**5 + 14956.34765625*cos(theta)**3 - 527.87109375*cos(theta))*sin(2*phi)

@torch.jit.script
def Yl13_m_minus_1(theta, phi):
    return 0.153658381323621*(1.0 - cos(theta)**2)**0.5*(16504.8583984375*cos(theta)**12 - 43572.826171875*cos(theta)**10 + 42625.5908203125*cos(theta)**8 - 18944.70703125*cos(theta)**6 + 3739.0869140625*cos(theta)**4 - 263.935546875*cos(theta)**2 + 2.9326171875)*sin(phi)

@torch.jit.script
def Yl13_m0(theta, phi):
    return 1860.99583201813*cos(theta)**13 - 5806.30699589657*cos(theta)**11 + 6942.32358205025*cos(theta)**9 - 3967.04204688585*cos(theta)**7 + 1096.15635506056*cos(theta)**5 - 128.959571183596*cos(theta)**3 + 4.29865237278653*cos(theta)

@torch.jit.script
def Yl13_m1(theta, phi):
    return 0.153658381323621*(1.0 - cos(theta)**2)**0.5*(16504.8583984375*cos(theta)**12 - 43572.826171875*cos(theta)**10 + 42625.5908203125*cos(theta)**8 - 18944.70703125*cos(theta)**6 + 3739.0869140625*cos(theta)**4 - 263.935546875*cos(theta)**2 + 2.9326171875)*cos(phi)

@torch.jit.script
def Yl13_m2(theta, phi):
    return 0.0114530195317401*(1.0 - cos(theta)**2)*(198058.30078125*cos(theta)**11 - 435728.26171875*cos(theta)**9 + 341004.7265625*cos(theta)**7 - 113668.2421875*cos(theta)**5 + 14956.34765625*cos(theta)**3 - 527.87109375*cos(theta))*cos(2*phi)

@torch.jit.script
def Yl13_m3(theta, phi):
    return 0.000863303829622583*(1.0 - cos(theta)**2)**1.5*(2178641.30859375*cos(theta)**10 - 3921554.35546875*cos(theta)**8 + 2387033.0859375*cos(theta)**6 - 568341.2109375*cos(theta)**4 + 44869.04296875*cos(theta)**2 - 527.87109375)*cos(3*phi)

@torch.jit.script
def Yl13_m4(theta, phi):
    return 6.62123812058377e-5*(1.0 - cos(theta)**2)**2*(21786413.0859375*cos(theta)**9 - 31372434.84375*cos(theta)**7 + 14322198.515625*cos(theta)**5 - 2273364.84375*cos(theta)**3 + 89738.0859375*cos(theta))*cos(4*phi)

@torch.jit.script
def Yl13_m5(theta, phi):
    return 5.2021359721285e-6*(1.0 - cos(theta)**2)**2.5*(196077717.773438*cos(theta)**8 - 219607043.90625*cos(theta)**6 + 71610992.578125*cos(theta)**4 - 6820094.53125*cos(theta)**2 + 89738.0859375)*cos(5*phi)

@torch.jit.script
def Yl13_m6(theta, phi):
    return 4.21948945157073e-7*(1.0 - cos(theta)**2)**3*(1568621742.1875*cos(theta)**7 - 1317642263.4375*cos(theta)**5 + 286443970.3125*cos(theta)**3 - 13640189.0625*cos(theta))*cos(6*phi)

@torch.jit.script
def Yl13_m7(theta, phi):
    return 3.5661194627771e-8*(1.0 - cos(theta)**2)**3.5*(10980352195.3125*cos(theta)**6 - 6588211317.1875*cos(theta)**4 + 859331910.9375*cos(theta)**2 - 13640189.0625)*cos(7*phi)

@torch.jit.script
def Yl13_m8(theta, phi):
    return 3.17695172143292e-9*(1.0 - cos(theta)**2)**4*(65882113171.875*cos(theta)**5 - 26352845268.75*cos(theta)**3 + 1718663821.875*cos(theta))*cos(8*phi)

@torch.jit.script
def Yl13_m9(theta, phi):
    return 3.02910461422567e-10*(1.0 - cos(theta)**2)**4.5*(329410565859.375*cos(theta)**4 - 79058535806.25*cos(theta)**2 + 1718663821.875)*cos(9*phi)

@torch.jit.script
def Yl13_m10(theta, phi):
    return 3.15805986876424e-11*(1.0 - cos(theta)**2)**5*(1317642263437.5*cos(theta)**3 - 158117071612.5*cos(theta))*cos(10*phi)

@torch.jit.script
def Yl13_m11(theta, phi):
    return 3.72180924766049e-12*(1.0 - cos(theta)**2)**5.5*(3952926790312.5*cos(theta)**2 - 158117071612.5)*cos(11*phi)

@torch.jit.script
def Yl13_m12(theta, phi):
    return 4.16119315354964*(1.0 - cos(theta)**2)**6*cos(12*phi)*cos(theta)

@torch.jit.script
def Yl13_m13(theta, phi):
    return 0.816077118837628*(1.0 - cos(theta)**2)**6.5*cos(13*phi)

@torch.jit.script
def Yl14_m_minus_14(theta, phi):
    return 0.830522083064524*(1.0 - cos(theta)**2)**7*sin(14*phi)

@torch.jit.script
def Yl14_m_minus_13(theta, phi):
    return 4.39470978027212*(1.0 - cos(theta)**2)**6.5*sin(13*phi)*cos(theta)

@torch.jit.script
def Yl14_m_minus_12(theta, phi):
    return 1.51291507116349e-13*(1.0 - cos(theta)**2)**6*(106729023338438.0*cos(theta)**2 - 3952926790312.5)*sin(12*phi)

@torch.jit.script
def Yl14_m_minus_11(theta, phi):
    return 1.33617041195793e-12*(1.0 - cos(theta)**2)**5.5*(35576341112812.5*cos(theta)**3 - 3952926790312.5*cos(theta))*sin(11*phi)

@torch.jit.script
def Yl14_m_minus_10(theta, phi):
    return 1.33617041195793e-11*(1.0 - cos(theta)**2)**5*(8894085278203.13*cos(theta)**4 - 1976463395156.25*cos(theta)**2 + 39529267903.125)*sin(10*phi)

@torch.jit.script
def Yl14_m_minus_9(theta, phi):
    return 1.46370135060066e-10*(1.0 - cos(theta)**2)**4.5*(1778817055640.63*cos(theta)**5 - 658821131718.75*cos(theta)**3 + 39529267903.125*cos(theta))*sin(9*phi)

@torch.jit.script
def Yl14_m_minus_8(theta, phi):
    return 1.71945976061531e-9*(1.0 - cos(theta)**2)**4*(296469509273.438*cos(theta)**6 - 164705282929.688*cos(theta)**4 + 19764633951.5625*cos(theta)**2 - 286443970.3125)*sin(8*phi)

@torch.jit.script
def Yl14_m_minus_7(theta, phi):
    return 2.13379344766496e-8*(1.0 - cos(theta)**2)**3.5*(42352787039.0625*cos(theta)**7 - 32941056585.9375*cos(theta)**5 + 6588211317.1875*cos(theta)**3 - 286443970.3125*cos(theta))*sin(7*phi)

@torch.jit.script
def Yl14_m_minus_6(theta, phi):
    return 2.76571240765567e-7*(1.0 - cos(theta)**2)**3*(5294098379.88281*cos(theta)**8 - 5490176097.65625*cos(theta)**6 + 1647052829.29688*cos(theta)**4 - 143221985.15625*cos(theta)**2 + 1705023.6328125)*sin(6*phi)

@torch.jit.script
def Yl14_m_minus_5(theta, phi):
    return 3.71059256983961e-6*(1.0 - cos(theta)**2)**2.5*(588233153.320313*cos(theta)**9 - 784310871.09375*cos(theta)**7 + 329410565.859375*cos(theta)**5 - 47740661.71875*cos(theta)**3 + 1705023.6328125*cos(theta))*sin(5*phi)

@torch.jit.script
def Yl14_m_minus_4(theta, phi):
    return 5.11469888818129e-5*(1.0 - cos(theta)**2)**2*(58823315.3320313*cos(theta)**10 - 98038858.8867188*cos(theta)**8 + 54901760.9765625*cos(theta)**6 - 11935165.4296875*cos(theta)**4 + 852511.81640625*cos(theta)**2 - 8973.80859375)*sin(4*phi)

@torch.jit.script
def Yl14_m_minus_3(theta, phi):
    return 0.000719701928156307*(1.0 - cos(theta)**2)**1.5*(5347574.12109375*cos(theta)**11 - 10893206.5429688*cos(theta)**9 + 7843108.7109375*cos(theta)**7 - 2387033.0859375*cos(theta)**5 + 284170.60546875*cos(theta)**3 - 8973.80859375*cos(theta))*sin(3*phi)

@torch.jit.script
def Yl14_m_minus_2(theta, phi):
    return 0.0102793996196251*(1.0 - cos(theta)**2)*(445631.176757813*cos(theta)**12 - 1089320.65429688*cos(theta)**10 + 980388.588867188*cos(theta)**8 - 397838.84765625*cos(theta)**6 + 71042.6513671875*cos(theta)**4 - 4486.904296875*cos(theta)**2 + 43.9892578125)*sin(2*phi)

@torch.jit.script
def Yl14_m_minus_1(theta, phi):
    return 0.148251609638173*(1.0 - cos(theta)**2)**0.5*(34279.3212890625*cos(theta)**13 - 99029.150390625*cos(theta)**11 + 108932.065429688*cos(theta)**9 - 56834.12109375*cos(theta)**7 + 14208.5302734375*cos(theta)**5 - 1495.634765625*cos(theta)**3 + 43.9892578125*cos(theta))*sin(phi)

@torch.jit.script
def Yl14_m0(theta, phi):
    return 3719.61718745389*cos(theta)**14 - 12536.487557715*cos(theta)**12 + 16548.1635761838*cos(theta)**10 - 10792.2805931633*cos(theta)**8 + 3597.42686438778*cos(theta)**6 - 568.014768061228*cos(theta)**4 + 33.4126334153663*cos(theta)**2 - 0.318215556336822

@torch.jit.script
def Yl14_m1(theta, phi):
    return 0.148251609638173*(1.0 - cos(theta)**2)**0.5*(34279.3212890625*cos(theta)**13 - 99029.150390625*cos(theta)**11 + 108932.065429688*cos(theta)**9 - 56834.12109375*cos(theta)**7 + 14208.5302734375*cos(theta)**5 - 1495.634765625*cos(theta)**3 + 43.9892578125*cos(theta))*cos(phi)

@torch.jit.script
def Yl14_m2(theta, phi):
    return 0.0102793996196251*(1.0 - cos(theta)**2)*(445631.176757813*cos(theta)**12 - 1089320.65429688*cos(theta)**10 + 980388.588867188*cos(theta)**8 - 397838.84765625*cos(theta)**6 + 71042.6513671875*cos(theta)**4 - 4486.904296875*cos(theta)**2 + 43.9892578125)*cos(2*phi)

@torch.jit.script
def Yl14_m3(theta, phi):
    return 0.000719701928156307*(1.0 - cos(theta)**2)**1.5*(5347574.12109375*cos(theta)**11 - 10893206.5429688*cos(theta)**9 + 7843108.7109375*cos(theta)**7 - 2387033.0859375*cos(theta)**5 + 284170.60546875*cos(theta)**3 - 8973.80859375*cos(theta))*cos(3*phi)

@torch.jit.script
def Yl14_m4(theta, phi):
    return 5.11469888818129e-5*(1.0 - cos(theta)**2)**2*(58823315.3320313*cos(theta)**10 - 98038858.8867188*cos(theta)**8 + 54901760.9765625*cos(theta)**6 - 11935165.4296875*cos(theta)**4 + 852511.81640625*cos(theta)**2 - 8973.80859375)*cos(4*phi)

@torch.jit.script
def Yl14_m5(theta, phi):
    return 3.71059256983961e-6*(1.0 - cos(theta)**2)**2.5*(588233153.320313*cos(theta)**9 - 784310871.09375*cos(theta)**7 + 329410565.859375*cos(theta)**5 - 47740661.71875*cos(theta)**3 + 1705023.6328125*cos(theta))*cos(5*phi)

@torch.jit.script
def Yl14_m6(theta, phi):
    return 2.76571240765567e-7*(1.0 - cos(theta)**2)**3*(5294098379.88281*cos(theta)**8 - 5490176097.65625*cos(theta)**6 + 1647052829.29688*cos(theta)**4 - 143221985.15625*cos(theta)**2 + 1705023.6328125)*cos(6*phi)

@torch.jit.script
def Yl14_m7(theta, phi):
    return 2.13379344766496e-8*(1.0 - cos(theta)**2)**3.5*(42352787039.0625*cos(theta)**7 - 32941056585.9375*cos(theta)**5 + 6588211317.1875*cos(theta)**3 - 286443970.3125*cos(theta))*cos(7*phi)

@torch.jit.script
def Yl14_m8(theta, phi):
    return 1.71945976061531e-9*(1.0 - cos(theta)**2)**4*(296469509273.438*cos(theta)**6 - 164705282929.688*cos(theta)**4 + 19764633951.5625*cos(theta)**2 - 286443970.3125)*cos(8*phi)

@torch.jit.script
def Yl14_m9(theta, phi):
    return 1.46370135060066e-10*(1.0 - cos(theta)**2)**4.5*(1778817055640.63*cos(theta)**5 - 658821131718.75*cos(theta)**3 + 39529267903.125*cos(theta))*cos(9*phi)

@torch.jit.script
def Yl14_m10(theta, phi):
    return 1.33617041195793e-11*(1.0 - cos(theta)**2)**5*(8894085278203.13*cos(theta)**4 - 1976463395156.25*cos(theta)**2 + 39529267903.125)*cos(10*phi)

@torch.jit.script
def Yl14_m11(theta, phi):
    return 1.33617041195793e-12*(1.0 - cos(theta)**2)**5.5*(35576341112812.5*cos(theta)**3 - 3952926790312.5*cos(theta))*cos(11*phi)

@torch.jit.script
def Yl14_m12(theta, phi):
    return 1.51291507116349e-13*(1.0 - cos(theta)**2)**6*(106729023338438.0*cos(theta)**2 - 3952926790312.5)*cos(12*phi)

@torch.jit.script
def Yl14_m13(theta, phi):
    return 4.39470978027212*(1.0 - cos(theta)**2)**6.5*cos(13*phi)*cos(theta)

@torch.jit.script
def Yl14_m14(theta, phi):
    return 0.830522083064524*(1.0 - cos(theta)**2)**7*cos(14*phi)

@torch.jit.script
def Yl15_m_minus_15(theta, phi):
    return 0.844250650857373*(1.0 - cos(theta)**2)**7.5*sin(15*phi)

@torch.jit.script
def Yl15_m_minus_14(theta, phi):
    return 4.62415125663001*(1.0 - cos(theta)**2)**7*sin(14*phi)*cos(theta)

@torch.jit.script
def Yl15_m_minus_13(theta, phi):
    return 5.68899431025918e-15*(1.0 - cos(theta)**2)**6.5*(3.09514167681469e+15*cos(theta)**2 - 106729023338438.0)*sin(13*phi)

@torch.jit.script
def Yl15_m_minus_12(theta, phi):
    return 5.21404941098716e-14*(1.0 - cos(theta)**2)**6*(1.03171389227156e+15*cos(theta)**3 - 106729023338438.0*cos(theta))*sin(12*phi)

@torch.jit.script
def Yl15_m_minus_11(theta, phi):
    return 5.4185990958026e-13*(1.0 - cos(theta)**2)**5.5*(257928473067891.0*cos(theta)**4 - 53364511669218.8*cos(theta)**2 + 988231697578.125)*sin(11*phi)

@torch.jit.script
def Yl15_m_minus_10(theta, phi):
    return 6.17815352749854e-12*(1.0 - cos(theta)**2)**5*(51585694613578.1*cos(theta)**5 - 17788170556406.3*cos(theta)**3 + 988231697578.125*cos(theta))*sin(10*phi)

@torch.jit.script
def Yl15_m_minus_9(theta, phi):
    return 7.56666184747369e-11*(1.0 - cos(theta)**2)**4.5*(8597615768929.69*cos(theta)**6 - 4447042639101.56*cos(theta)**4 + 494115848789.063*cos(theta)**2 - 6588211317.1875)*sin(9*phi)

@torch.jit.script
def Yl15_m_minus_8(theta, phi):
    return 9.80751467720255e-10*(1.0 - cos(theta)**2)**4*(1228230824132.81*cos(theta)**7 - 889408527820.313*cos(theta)**5 + 164705282929.688*cos(theta)**3 - 6588211317.1875*cos(theta))*sin(8*phi)

@torch.jit.script
def Yl15_m_minus_7(theta, phi):
    return 1.33035601710264e-8*(1.0 - cos(theta)**2)**3.5*(153528853016.602*cos(theta)**8 - 148234754636.719*cos(theta)**6 + 41176320732.4219*cos(theta)**4 - 3294105658.59375*cos(theta)**2 + 35805496.2890625)*sin(7*phi)

@torch.jit.script
def Yl15_m_minus_6(theta, phi):
    return 1.87197684863824e-7*(1.0 - cos(theta)**2)**3*(17058761446.2891*cos(theta)**9 - 21176393519.5313*cos(theta)**7 + 8235264146.48438*cos(theta)**5 - 1098035219.53125*cos(theta)**3 + 35805496.2890625*cos(theta))*sin(6*phi)

@torch.jit.script
def Yl15_m_minus_5(theta, phi):
    return 2.71275217737612e-6*(1.0 - cos(theta)**2)**2.5*(1705876144.62891*cos(theta)**10 - 2647049189.94141*cos(theta)**8 + 1372544024.41406*cos(theta)**6 - 274508804.882813*cos(theta)**4 + 17902748.1445313*cos(theta)**2 - 170502.36328125)*sin(5*phi)

@torch.jit.script
def Yl15_m_minus_4(theta, phi):
    return 4.02366171874445e-5*(1.0 - cos(theta)**2)**2*(155079649.511719*cos(theta)**11 - 294116576.660156*cos(theta)**9 + 196077717.773438*cos(theta)**7 - 54901760.9765625*cos(theta)**5 + 5967582.71484375*cos(theta)**3 - 170502.36328125*cos(theta))*sin(4*phi)

@torch.jit.script
def Yl15_m_minus_3(theta, phi):
    return 0.000607559596001151*(1.0 - cos(theta)**2)**1.5*(12923304.1259766*cos(theta)**12 - 29411657.6660156*cos(theta)**10 + 24509714.7216797*cos(theta)**8 - 9150293.49609375*cos(theta)**6 + 1491895.67871094*cos(theta)**4 - 85251.181640625*cos(theta)**2 + 747.8173828125)*sin(3*phi)

@torch.jit.script
def Yl15_m_minus_2(theta, phi):
    return 0.00929387470704126*(1.0 - cos(theta)**2)*(994100.317382813*cos(theta)**13 - 2673787.06054688*cos(theta)**11 + 2723301.63574219*cos(theta)**9 - 1307184.78515625*cos(theta)**7 + 298379.135742188*cos(theta)**5 - 28417.060546875*cos(theta)**3 + 747.8173828125*cos(theta))*sin(2*phi)

@torch.jit.script
def Yl15_m_minus_1(theta, phi):
    return 0.143378915753688*(1.0 - cos(theta)**2)**0.5*(71007.1655273438*cos(theta)**14 - 222815.588378906*cos(theta)**12 + 272330.163574219*cos(theta)**10 - 163398.098144531*cos(theta)**8 + 49729.8559570313*cos(theta)**6 - 7104.26513671875*cos(theta)**4 + 373.90869140625*cos(theta)**2 - 3.14208984375)*sin(phi)

@torch.jit.script
def Yl15_m0(theta, phi):
    return 7435.10031825349*cos(theta)**15 - 26920.1908074695*cos(theta)**13 + 38884.7200552338*cos(theta)**11 - 28515.4613738381*cos(theta)**9 + 11158.2240158497*cos(theta)**7 - 2231.64480316994*cos(theta)**5 + 195.758316067539*cos(theta)**3 - 4.93508359834131*cos(theta)

@torch.jit.script
def Yl15_m1(theta, phi):
    return 0.143378915753688*(1.0 - cos(theta)**2)**0.5*(71007.1655273438*cos(theta)**14 - 222815.588378906*cos(theta)**12 + 272330.163574219*cos(theta)**10 - 163398.098144531*cos(theta)**8 + 49729.8559570313*cos(theta)**6 - 7104.26513671875*cos(theta)**4 + 373.90869140625*cos(theta)**2 - 3.14208984375)*cos(phi)

@torch.jit.script
def Yl15_m2(theta, phi):
    return 0.00929387470704126*(1.0 - cos(theta)**2)*(994100.317382813*cos(theta)**13 - 2673787.06054688*cos(theta)**11 + 2723301.63574219*cos(theta)**9 - 1307184.78515625*cos(theta)**7 + 298379.135742188*cos(theta)**5 - 28417.060546875*cos(theta)**3 + 747.8173828125*cos(theta))*cos(2*phi)

@torch.jit.script
def Yl15_m3(theta, phi):
    return 0.000607559596001151*(1.0 - cos(theta)**2)**1.5*(12923304.1259766*cos(theta)**12 - 29411657.6660156*cos(theta)**10 + 24509714.7216797*cos(theta)**8 - 9150293.49609375*cos(theta)**6 + 1491895.67871094*cos(theta)**4 - 85251.181640625*cos(theta)**2 + 747.8173828125)*cos(3*phi)

@torch.jit.script
def Yl15_m4(theta, phi):
    return 4.02366171874445e-5*(1.0 - cos(theta)**2)**2*(155079649.511719*cos(theta)**11 - 294116576.660156*cos(theta)**9 + 196077717.773438*cos(theta)**7 - 54901760.9765625*cos(theta)**5 + 5967582.71484375*cos(theta)**3 - 170502.36328125*cos(theta))*cos(4*phi)

@torch.jit.script
def Yl15_m5(theta, phi):
    return 2.71275217737612e-6*(1.0 - cos(theta)**2)**2.5*(1705876144.62891*cos(theta)**10 - 2647049189.94141*cos(theta)**8 + 1372544024.41406*cos(theta)**6 - 274508804.882813*cos(theta)**4 + 17902748.1445313*cos(theta)**2 - 170502.36328125)*cos(5*phi)

@torch.jit.script
def Yl15_m6(theta, phi):
    return 1.87197684863824e-7*(1.0 - cos(theta)**2)**3*(17058761446.2891*cos(theta)**9 - 21176393519.5313*cos(theta)**7 + 8235264146.48438*cos(theta)**5 - 1098035219.53125*cos(theta)**3 + 35805496.2890625*cos(theta))*cos(6*phi)

@torch.jit.script
def Yl15_m7(theta, phi):
    return 1.33035601710264e-8*(1.0 - cos(theta)**2)**3.5*(153528853016.602*cos(theta)**8 - 148234754636.719*cos(theta)**6 + 41176320732.4219*cos(theta)**4 - 3294105658.59375*cos(theta)**2 + 35805496.2890625)*cos(7*phi)

@torch.jit.script
def Yl15_m8(theta, phi):
    return 9.80751467720255e-10*(1.0 - cos(theta)**2)**4*(1228230824132.81*cos(theta)**7 - 889408527820.313*cos(theta)**5 + 164705282929.688*cos(theta)**3 - 6588211317.1875*cos(theta))*cos(8*phi)

@torch.jit.script
def Yl15_m9(theta, phi):
    return 7.56666184747369e-11*(1.0 - cos(theta)**2)**4.5*(8597615768929.69*cos(theta)**6 - 4447042639101.56*cos(theta)**4 + 494115848789.063*cos(theta)**2 - 6588211317.1875)*cos(9*phi)

@torch.jit.script
def Yl15_m10(theta, phi):
    return 6.17815352749854e-12*(1.0 - cos(theta)**2)**5*(51585694613578.1*cos(theta)**5 - 17788170556406.3*cos(theta)**3 + 988231697578.125*cos(theta))*cos(10*phi)

@torch.jit.script
def Yl15_m11(theta, phi):
    return 5.4185990958026e-13*(1.0 - cos(theta)**2)**5.5*(257928473067891.0*cos(theta)**4 - 53364511669218.8*cos(theta)**2 + 988231697578.125)*cos(11*phi)

@torch.jit.script
def Yl15_m12(theta, phi):
    return 5.21404941098716e-14*(1.0 - cos(theta)**2)**6*(1.03171389227156e+15*cos(theta)**3 - 106729023338438.0*cos(theta))*cos(12*phi)

@torch.jit.script
def Yl15_m13(theta, phi):
    return 5.68899431025918e-15*(1.0 - cos(theta)**2)**6.5*(3.09514167681469e+15*cos(theta)**2 - 106729023338438.0)*cos(13*phi)

@torch.jit.script
def Yl15_m14(theta, phi):
    return 4.62415125663001*(1.0 - cos(theta)**2)**7*cos(14*phi)*cos(theta)

@torch.jit.script
def Yl15_m15(theta, phi):
    return 0.844250650857373*(1.0 - cos(theta)**2)**7.5*cos(15*phi)

@torch.jit.script
def Yl16_m_minus_16(theta, phi):
    return 0.857340588838025*(1.0 - cos(theta)**2)**8*sin(16*phi)

@torch.jit.script
def Yl16_m_minus_15(theta, phi):
    return 4.84985075323068*(1.0 - cos(theta)**2)**7.5*sin(15*phi)*cos(theta)

@torch.jit.script
def Yl16_m_minus_14(theta, phi):
    return 1.98999505000411e-16*(1.0 - cos(theta)**2)**7*(9.59493919812553e+16*cos(theta)**2 - 3.09514167681469e+15)*sin(14*phi)

@torch.jit.script
def Yl16_m_minus_13(theta, phi):
    return 1.8878750671421e-15*(1.0 - cos(theta)**2)**6.5*(3.19831306604184e+16*cos(theta)**3 - 3.09514167681469e+15*cos(theta))*sin(13*phi)

@torch.jit.script
def Yl16_m_minus_12(theta, phi):
    return 2.03330367436807e-14*(1.0 - cos(theta)**2)**6*(7.99578266510461e+15*cos(theta)**4 - 1.54757083840734e+15*cos(theta)**2 + 26682255834609.4)*sin(12*phi)

@torch.jit.script
def Yl16_m_minus_11(theta, phi):
    return 2.40583735216622e-13*(1.0 - cos(theta)**2)**5.5*(1.59915653302092e+15*cos(theta)**5 - 515856946135781.0*cos(theta)**3 + 26682255834609.4*cos(theta))*sin(11*phi)

@torch.jit.script
def Yl16_m_minus_10(theta, phi):
    return 3.06213103106751e-12*(1.0 - cos(theta)**2)**5*(266526088836820.0*cos(theta)**6 - 128964236533945.0*cos(theta)**4 + 13341127917304.7*cos(theta)**2 - 164705282929.688)*sin(10*phi)

@torch.jit.script
def Yl16_m_minus_9(theta, phi):
    return 4.1310406124361e-11*(1.0 - cos(theta)**2)**4.5*(38075155548117.2*cos(theta)**7 - 25792847306789.1*cos(theta)**5 + 4447042639101.56*cos(theta)**3 - 164705282929.688*cos(theta))*sin(9*phi)

@torch.jit.script
def Yl16_m_minus_8(theta, phi):
    return 5.84217366082119e-10*(1.0 - cos(theta)**2)**4*(4759394443514.65*cos(theta)**8 - 4298807884464.84*cos(theta)**6 + 1111760659775.39*cos(theta)**4 - 82352641464.8438*cos(theta)**2 + 823526414.648438)*sin(8*phi)

@torch.jit.script
def Yl16_m_minus_7(theta, phi):
    return 8.58620667464373e-9*(1.0 - cos(theta)**2)**3.5*(528821604834.961*cos(theta)**9 - 614115412066.406*cos(theta)**7 + 222352131955.078*cos(theta)**5 - 27450880488.2813*cos(theta)**3 + 823526414.648438*cos(theta))*sin(7*phi)

@torch.jit.script
def Yl16_m_minus_6(theta, phi):
    return 1.30216271501415e-7*(1.0 - cos(theta)**2)**3*(52882160483.4961*cos(theta)**10 - 76764426508.3008*cos(theta)**8 + 37058688659.1797*cos(theta)**6 - 6862720122.07031*cos(theta)**4 + 411763207.324219*cos(theta)**2 - 3580549.62890625)*sin(6*phi)

@torch.jit.script
def Yl16_m_minus_5(theta, phi):
    return 2.02568978918854e-6*(1.0 - cos(theta)**2)**2.5*(4807469134.86328*cos(theta)**11 - 8529380723.14453*cos(theta)**9 + 5294098379.88281*cos(theta)**7 - 1372544024.41406*cos(theta)**5 + 137254402.441406*cos(theta)**3 - 3580549.62890625*cos(theta))*sin(5*phi)

@torch.jit.script
def Yl16_m_minus_4(theta, phi):
    return 3.21568284933344e-5*(1.0 - cos(theta)**2)**2*(400622427.905273*cos(theta)**12 - 852938072.314453*cos(theta)**10 + 661762297.485352*cos(theta)**8 - 228757337.402344*cos(theta)**6 + 34313600.6103516*cos(theta)**4 - 1790274.81445313*cos(theta)**2 + 14208.5302734375)*sin(4*phi)

@torch.jit.script
def Yl16_m_minus_3(theta, phi):
    return 0.000518513279362185*(1.0 - cos(theta)**2)**1.5*(30817109.8388672*cos(theta)**13 - 77539824.7558594*cos(theta)**11 + 73529144.1650391*cos(theta)**9 - 32679619.6289063*cos(theta)**7 + 6862720.12207031*cos(theta)**5 - 596758.271484375*cos(theta)**3 + 14208.5302734375*cos(theta))*sin(3*phi)

@torch.jit.script
def Yl16_m_minus_2(theta, phi):
    return 0.00845669566395355*(1.0 - cos(theta)**2)*(2201222.13134766*cos(theta)**14 - 6461652.06298828*cos(theta)**12 + 7352914.41650391*cos(theta)**10 - 4084952.45361328*cos(theta)**8 + 1143786.68701172*cos(theta)**6 - 149189.567871094*cos(theta)**4 + 7104.26513671875*cos(theta)**2 - 53.41552734375)*sin(2*phi)

@torch.jit.script
def Yl16_m_minus_1(theta, phi):
    return 0.138957689313105*(1.0 - cos(theta)**2)**0.5*(146748.142089844*cos(theta)**15 - 497050.158691406*cos(theta)**13 + 668446.765136719*cos(theta)**11 - 453883.605957031*cos(theta)**9 + 163398.098144531*cos(theta)**7 - 29837.9135742188*cos(theta)**5 + 2368.08837890625*cos(theta)**3 - 53.41552734375*cos(theta))*sin(phi)

@torch.jit.script
def Yl16_m0(theta, phi):
    return 14862.9380228203*cos(theta)**16 - 57533.9536367237*cos(theta)**14 + 90268.7893265838*cos(theta)**12 - 73552.3468586979*cos(theta)**10 + 33098.5560864141*cos(theta)**8 - 8058.77887321386*cos(theta)**6 + 959.378437287364*cos(theta)**4 - 43.2802302535653*cos(theta)**2 + 0.318236987158568

@torch.jit.script
def Yl16_m1(theta, phi):
    return 0.138957689313105*(1.0 - cos(theta)**2)**0.5*(146748.142089844*cos(theta)**15 - 497050.158691406*cos(theta)**13 + 668446.765136719*cos(theta)**11 - 453883.605957031*cos(theta)**9 + 163398.098144531*cos(theta)**7 - 29837.9135742188*cos(theta)**5 + 2368.08837890625*cos(theta)**3 - 53.41552734375*cos(theta))*cos(phi)

@torch.jit.script
def Yl16_m2(theta, phi):
    return 0.00845669566395355*(1.0 - cos(theta)**2)*(2201222.13134766*cos(theta)**14 - 6461652.06298828*cos(theta)**12 + 7352914.41650391*cos(theta)**10 - 4084952.45361328*cos(theta)**8 + 1143786.68701172*cos(theta)**6 - 149189.567871094*cos(theta)**4 + 7104.26513671875*cos(theta)**2 - 53.41552734375)*cos(2*phi)

@torch.jit.script
def Yl16_m3(theta, phi):
    return 0.000518513279362185*(1.0 - cos(theta)**2)**1.5*(30817109.8388672*cos(theta)**13 - 77539824.7558594*cos(theta)**11 + 73529144.1650391*cos(theta)**9 - 32679619.6289063*cos(theta)**7 + 6862720.12207031*cos(theta)**5 - 596758.271484375*cos(theta)**3 + 14208.5302734375*cos(theta))*cos(3*phi)

@torch.jit.script
def Yl16_m4(theta, phi):
    return 3.21568284933344e-5*(1.0 - cos(theta)**2)**2*(400622427.905273*cos(theta)**12 - 852938072.314453*cos(theta)**10 + 661762297.485352*cos(theta)**8 - 228757337.402344*cos(theta)**6 + 34313600.6103516*cos(theta)**4 - 1790274.81445313*cos(theta)**2 + 14208.5302734375)*cos(4*phi)

@torch.jit.script
def Yl16_m5(theta, phi):
    return 2.02568978918854e-6*(1.0 - cos(theta)**2)**2.5*(4807469134.86328*cos(theta)**11 - 8529380723.14453*cos(theta)**9 + 5294098379.88281*cos(theta)**7 - 1372544024.41406*cos(theta)**5 + 137254402.441406*cos(theta)**3 - 3580549.62890625*cos(theta))*cos(5*phi)

@torch.jit.script
def Yl16_m6(theta, phi):
    return 1.30216271501415e-7*(1.0 - cos(theta)**2)**3*(52882160483.4961*cos(theta)**10 - 76764426508.3008*cos(theta)**8 + 37058688659.1797*cos(theta)**6 - 6862720122.07031*cos(theta)**4 + 411763207.324219*cos(theta)**2 - 3580549.62890625)*cos(6*phi)

@torch.jit.script
def Yl16_m7(theta, phi):
    return 8.58620667464373e-9*(1.0 - cos(theta)**2)**3.5*(528821604834.961*cos(theta)**9 - 614115412066.406*cos(theta)**7 + 222352131955.078*cos(theta)**5 - 27450880488.2813*cos(theta)**3 + 823526414.648438*cos(theta))*cos(7*phi)

@torch.jit.script
def Yl16_m8(theta, phi):
    return 5.84217366082119e-10*(1.0 - cos(theta)**2)**4*(4759394443514.65*cos(theta)**8 - 4298807884464.84*cos(theta)**6 + 1111760659775.39*cos(theta)**4 - 82352641464.8438*cos(theta)**2 + 823526414.648438)*cos(8*phi)

@torch.jit.script
def Yl16_m9(theta, phi):
    return 4.1310406124361e-11*(1.0 - cos(theta)**2)**4.5*(38075155548117.2*cos(theta)**7 - 25792847306789.1*cos(theta)**5 + 4447042639101.56*cos(theta)**3 - 164705282929.688*cos(theta))*cos(9*phi)

@torch.jit.script
def Yl16_m10(theta, phi):
    return 3.06213103106751e-12*(1.0 - cos(theta)**2)**5*(266526088836820.0*cos(theta)**6 - 128964236533945.0*cos(theta)**4 + 13341127917304.7*cos(theta)**2 - 164705282929.688)*cos(10*phi)

@torch.jit.script
def Yl16_m11(theta, phi):
    return 2.40583735216622e-13*(1.0 - cos(theta)**2)**5.5*(1.59915653302092e+15*cos(theta)**5 - 515856946135781.0*cos(theta)**3 + 26682255834609.4*cos(theta))*cos(11*phi)

@torch.jit.script
def Yl16_m12(theta, phi):
    return 2.03330367436807e-14*(1.0 - cos(theta)**2)**6*(7.99578266510461e+15*cos(theta)**4 - 1.54757083840734e+15*cos(theta)**2 + 26682255834609.4)*cos(12*phi)

@torch.jit.script
def Yl16_m13(theta, phi):
    return 1.8878750671421e-15*(1.0 - cos(theta)**2)**6.5*(3.19831306604184e+16*cos(theta)**3 - 3.09514167681469e+15*cos(theta))*cos(13*phi)

@torch.jit.script
def Yl16_m14(theta, phi):
    return 1.98999505000411e-16*(1.0 - cos(theta)**2)**7*(9.59493919812553e+16*cos(theta)**2 - 3.09514167681469e+15)*cos(14*phi)

@torch.jit.script
def Yl16_m15(theta, phi):
    return 4.84985075323068*(1.0 - cos(theta)**2)**7.5*cos(15*phi)*cos(theta)

@torch.jit.script
def Yl16_m16(theta, phi):
    return 0.857340588838025*(1.0 - cos(theta)**2)**8*cos(16*phi)

@torch.jit.script
def Yl17_m_minus_17(theta, phi):
    return 0.869857171920628*(1.0 - cos(theta)**2)**8.5*sin(17*phi)

@torch.jit.script
def Yl17_m_minus_16(theta, phi):
    return 5.07209532485536*(1.0 - cos(theta)**2)**8*sin(16*phi)*cos(theta)

@torch.jit.script
def Yl17_m_minus_15(theta, phi):
    return 6.50688621401289e-18*(1.0 - cos(theta)**2)**7.5*(3.16632993538143e+18*cos(theta)**2 - 9.59493919812553e+16)*sin(15*phi)

@torch.jit.script
def Yl17_m_minus_14(theta, phi):
    return 6.37542041547274e-17*(1.0 - cos(theta)**2)**7*(1.05544331179381e+18*cos(theta)**3 - 9.59493919812553e+16*cos(theta))*sin(14*phi)

@torch.jit.script
def Yl17_m_minus_13(theta, phi):
    return 7.09936771746562e-16*(1.0 - cos(theta)**2)**6.5*(2.63860827948452e+17*cos(theta)**4 - 4.79746959906277e+16*cos(theta)**2 + 773785419203672.0)*sin(13*phi)

@torch.jit.script
def Yl17_m_minus_12(theta, phi):
    return 8.69491420208903e-15*(1.0 - cos(theta)**2)**6*(5.27721655896904e+16*cos(theta)**5 - 1.59915653302092e+16*cos(theta)**3 + 773785419203672.0*cos(theta))*sin(12*phi)

@torch.jit.script
def Yl17_m_minus_11(theta, phi):
    return 1.14693795555008e-13*(1.0 - cos(theta)**2)**5.5*(8.79536093161507e+15*cos(theta)**6 - 3.9978913325523e+15*cos(theta)**4 + 386892709601836.0*cos(theta)**2 - 4447042639101.56)*sin(11*phi)

@torch.jit.script
def Yl17_m_minus_10(theta, phi):
    return 1.60571313777011e-12*(1.0 - cos(theta)**2)**5*(1.25648013308787e+15*cos(theta)**7 - 799578266510461.0*cos(theta)**5 + 128964236533945.0*cos(theta)**3 - 4447042639101.56*cos(theta))*sin(10*phi)

@torch.jit.script
def Yl17_m_minus_9(theta, phi):
    return 2.35990671649205e-11*(1.0 - cos(theta)**2)**4.5*(157060016635983.0*cos(theta)**8 - 133263044418410.0*cos(theta)**6 + 32241059133486.3*cos(theta)**4 - 2223521319550.78*cos(theta)**2 + 20588160366.2109)*sin(9*phi)

@torch.jit.script
def Yl17_m_minus_8(theta, phi):
    return 3.60996311929549e-10*(1.0 - cos(theta)**2)**4*(17451112959553.7*cos(theta)**9 - 19037577774058.6*cos(theta)**7 + 6448211826697.27*cos(theta)**5 - 741173773183.594*cos(theta)**3 + 20588160366.2109*cos(theta))*sin(8*phi)

@torch.jit.script
def Yl17_m_minus_7(theta, phi):
    return 5.70785286308994e-9*(1.0 - cos(theta)**2)**3.5*(1745111295955.37*cos(theta)**10 - 2379697221757.32*cos(theta)**8 + 1074701971116.21*cos(theta)**6 - 185293443295.898*cos(theta)**4 + 10294080183.1055*cos(theta)**2 - 82352641.4648438)*sin(7*phi)

@torch.jit.script
def Yl17_m_minus_6(theta, phi):
    return 9.2741631735508e-8*(1.0 - cos(theta)**2)**3*(158646481450.488*cos(theta)**11 - 264410802417.48*cos(theta)**9 + 153528853016.602*cos(theta)**7 - 37058688659.1797*cos(theta)**5 + 3431360061.03516*cos(theta)**3 - 82352641.4648438*cos(theta))*sin(6*phi)

@torch.jit.script
def Yl17_m_minus_5(theta, phi):
    return 1.54073970252026e-6*(1.0 - cos(theta)**2)**2.5*(13220540120.874*cos(theta)**12 - 26441080241.748*cos(theta)**10 + 19191106627.0752*cos(theta)**8 - 6176448109.86328*cos(theta)**6 + 857840015.258789*cos(theta)**4 - 41176320.7324219*cos(theta)**2 + 298379.135742188)*sin(5*phi)

@torch.jit.script
def Yl17_m_minus_4(theta, phi):
    return 2.6056272673653e-5*(1.0 - cos(theta)**2)**2*(1016964624.68262*cos(theta)**13 - 2403734567.43164*cos(theta)**11 + 2132345180.78613*cos(theta)**9 - 882349729.980469*cos(theta)**7 + 171568003.051758*cos(theta)**5 - 13725440.2441406*cos(theta)**3 + 298379.135742188*cos(theta))*sin(4*phi)

@torch.jit.script
def Yl17_m_minus_3(theta, phi):
    return 0.000446772008544923*(1.0 - cos(theta)**2)**1.5*(72640330.3344727*cos(theta)**14 - 200311213.952637*cos(theta)**12 + 213234518.078613*cos(theta)**10 - 110293716.247559*cos(theta)**8 + 28594667.175293*cos(theta)**6 - 3431360.06103516*cos(theta)**4 + 149189.567871094*cos(theta)**2 - 1014.89501953125)*sin(3*phi)

@torch.jit.script
def Yl17_m_minus_2(theta, phi):
    return 0.00773831818199403*(1.0 - cos(theta)**2)*(4842688.68896484*cos(theta)**15 - 15408554.9194336*cos(theta)**13 + 19384956.1889648*cos(theta)**11 - 12254857.3608398*cos(theta)**9 + 4084952.45361328*cos(theta)**7 - 686272.012207031*cos(theta)**5 + 49729.8559570313*cos(theta)**3 - 1014.89501953125*cos(theta))*sin(2*phi)

@torch.jit.script
def Yl17_m_minus_1(theta, phi):
    return 0.134922187793101*(1.0 - cos(theta)**2)**0.5*(302668.043060303*cos(theta)**16 - 1100611.06567383*cos(theta)**14 + 1615413.01574707*cos(theta)**12 - 1225485.73608398*cos(theta)**10 + 510619.05670166*cos(theta)**8 - 114378.668701172*cos(theta)**6 + 12432.4639892578*cos(theta)**4 - 507.447509765625*cos(theta)**2 + 3.33847045898438)*sin(phi)

@torch.jit.script
def Yl17_m0(theta, phi):
    return 29713.0160510757*cos(theta)**17 - 122453.641907463*cos(theta)**15 + 207381.167746511*cos(theta)**13 - 185927.943496872*cos(theta)**11 + 94685.5267808142*cos(theta)**9 - 27269.4317128745*cos(theta)**7 + 4149.69613022003*cos(theta)**5 - 282.292253756465*cos(theta)**3 + 5.57155763993023*cos(theta)

@torch.jit.script
def Yl17_m1(theta, phi):
    return 0.134922187793101*(1.0 - cos(theta)**2)**0.5*(302668.043060303*cos(theta)**16 - 1100611.06567383*cos(theta)**14 + 1615413.01574707*cos(theta)**12 - 1225485.73608398*cos(theta)**10 + 510619.05670166*cos(theta)**8 - 114378.668701172*cos(theta)**6 + 12432.4639892578*cos(theta)**4 - 507.447509765625*cos(theta)**2 + 3.33847045898438)*cos(phi)

@torch.jit.script
def Yl17_m2(theta, phi):
    return 0.00773831818199403*(1.0 - cos(theta)**2)*(4842688.68896484*cos(theta)**15 - 15408554.9194336*cos(theta)**13 + 19384956.1889648*cos(theta)**11 - 12254857.3608398*cos(theta)**9 + 4084952.45361328*cos(theta)**7 - 686272.012207031*cos(theta)**5 + 49729.8559570313*cos(theta)**3 - 1014.89501953125*cos(theta))*cos(2*phi)

@torch.jit.script
def Yl17_m3(theta, phi):
    return 0.000446772008544923*(1.0 - cos(theta)**2)**1.5*(72640330.3344727*cos(theta)**14 - 200311213.952637*cos(theta)**12 + 213234518.078613*cos(theta)**10 - 110293716.247559*cos(theta)**8 + 28594667.175293*cos(theta)**6 - 3431360.06103516*cos(theta)**4 + 149189.567871094*cos(theta)**2 - 1014.89501953125)*cos(3*phi)

@torch.jit.script
def Yl17_m4(theta, phi):
    return 2.6056272673653e-5*(1.0 - cos(theta)**2)**2*(1016964624.68262*cos(theta)**13 - 2403734567.43164*cos(theta)**11 + 2132345180.78613*cos(theta)**9 - 882349729.980469*cos(theta)**7 + 171568003.051758*cos(theta)**5 - 13725440.2441406*cos(theta)**3 + 298379.135742188*cos(theta))*cos(4*phi)

@torch.jit.script
def Yl17_m5(theta, phi):
    return 1.54073970252026e-6*(1.0 - cos(theta)**2)**2.5*(13220540120.874*cos(theta)**12 - 26441080241.748*cos(theta)**10 + 19191106627.0752*cos(theta)**8 - 6176448109.86328*cos(theta)**6 + 857840015.258789*cos(theta)**4 - 41176320.7324219*cos(theta)**2 + 298379.135742188)*cos(5*phi)

@torch.jit.script
def Yl17_m6(theta, phi):
    return 9.2741631735508e-8*(1.0 - cos(theta)**2)**3*(158646481450.488*cos(theta)**11 - 264410802417.48*cos(theta)**9 + 153528853016.602*cos(theta)**7 - 37058688659.1797*cos(theta)**5 + 3431360061.03516*cos(theta)**3 - 82352641.4648438*cos(theta))*cos(6*phi)

@torch.jit.script
def Yl17_m7(theta, phi):
    return 5.70785286308994e-9*(1.0 - cos(theta)**2)**3.5*(1745111295955.37*cos(theta)**10 - 2379697221757.32*cos(theta)**8 + 1074701971116.21*cos(theta)**6 - 185293443295.898*cos(theta)**4 + 10294080183.1055*cos(theta)**2 - 82352641.4648438)*cos(7*phi)

@torch.jit.script
def Yl17_m8(theta, phi):
    return 3.60996311929549e-10*(1.0 - cos(theta)**2)**4*(17451112959553.7*cos(theta)**9 - 19037577774058.6*cos(theta)**7 + 6448211826697.27*cos(theta)**5 - 741173773183.594*cos(theta)**3 + 20588160366.2109*cos(theta))*cos(8*phi)

@torch.jit.script
def Yl17_m9(theta, phi):
    return 2.35990671649205e-11*(1.0 - cos(theta)**2)**4.5*(157060016635983.0*cos(theta)**8 - 133263044418410.0*cos(theta)**6 + 32241059133486.3*cos(theta)**4 - 2223521319550.78*cos(theta)**2 + 20588160366.2109)*cos(9*phi)

@torch.jit.script
def Yl17_m10(theta, phi):
    return 1.60571313777011e-12*(1.0 - cos(theta)**2)**5*(1.25648013308787e+15*cos(theta)**7 - 799578266510461.0*cos(theta)**5 + 128964236533945.0*cos(theta)**3 - 4447042639101.56*cos(theta))*cos(10*phi)

@torch.jit.script
def Yl17_m11(theta, phi):
    return 1.14693795555008e-13*(1.0 - cos(theta)**2)**5.5*(8.79536093161507e+15*cos(theta)**6 - 3.9978913325523e+15*cos(theta)**4 + 386892709601836.0*cos(theta)**2 - 4447042639101.56)*cos(11*phi)

@torch.jit.script
def Yl17_m12(theta, phi):
    return 8.69491420208903e-15*(1.0 - cos(theta)**2)**6*(5.27721655896904e+16*cos(theta)**5 - 1.59915653302092e+16*cos(theta)**3 + 773785419203672.0*cos(theta))*cos(12*phi)

@torch.jit.script
def Yl17_m13(theta, phi):
    return 7.09936771746562e-16*(1.0 - cos(theta)**2)**6.5*(2.63860827948452e+17*cos(theta)**4 - 4.79746959906277e+16*cos(theta)**2 + 773785419203672.0)*cos(13*phi)

@torch.jit.script
def Yl17_m14(theta, phi):
    return 6.37542041547274e-17*(1.0 - cos(theta)**2)**7*(1.05544331179381e+18*cos(theta)**3 - 9.59493919812553e+16*cos(theta))*cos(14*phi)

@torch.jit.script
def Yl17_m15(theta, phi):
    return 6.50688621401289e-18*(1.0 - cos(theta)**2)**7.5*(3.16632993538143e+18*cos(theta)**2 - 9.59493919812553e+16)*cos(15*phi)

@torch.jit.script
def Yl17_m16(theta, phi):
    return 5.07209532485536*(1.0 - cos(theta)**2)**8*cos(16*phi)*cos(theta)

@torch.jit.script
def Yl17_m17(theta, phi):
    return 0.869857171920628*(1.0 - cos(theta)**2)**8.5*cos(17*phi)

@torch.jit.script
def Yl18_m_minus_18(theta, phi):
    return 0.881855768678329*(1.0 - cos(theta)**2)**9*sin(18*phi)

@torch.jit.script
def Yl18_m_minus_17(theta, phi):
    return 5.29113461206997*(1.0 - cos(theta)**2)**8.5*sin(17*phi)*cos(theta)

@torch.jit.script
def Yl18_m_minus_16(theta, phi):
    return 1.99730147939357e-19*(1.0 - cos(theta)**2)**8*(1.1082154773835e+20*cos(theta)**2 - 3.16632993538143e+18)*sin(16*phi)

@torch.jit.script
def Yl18_m_minus_15(theta, phi):
    return 2.01717561545333e-18*(1.0 - cos(theta)**2)**7.5*(3.69405159127833e+19*cos(theta)**3 - 3.16632993538143e+18*cos(theta))*sin(15*phi)

@torch.jit.script
def Yl18_m_minus_14(theta, phi):
    return 2.31755833840811e-17*(1.0 - cos(theta)**2)**7*(9.23512897819582e+18*cos(theta)**4 - 1.58316496769071e+18*cos(theta)**2 + 2.39873479953138e+16)*sin(14*phi)

@torch.jit.script
def Yl18_m_minus_13(theta, phi):
    return 2.93150518387396e-16*(1.0 - cos(theta)**2)**6.5*(1.84702579563916e+18*cos(theta)**5 - 5.27721655896904e+17*cos(theta)**3 + 2.39873479953138e+16*cos(theta))*sin(13*phi)

@torch.jit.script
def Yl18_m_minus_12(theta, phi):
    return 3.9980400343329e-15*(1.0 - cos(theta)**2)**6*(3.07837632606527e+17*cos(theta)**6 - 1.31930413974226e+17*cos(theta)**4 + 1.19936739976569e+16*cos(theta)**2 - 128964236533945.0)*sin(12*phi)

@torch.jit.script
def Yl18_m_minus_11(theta, phi):
    return 5.79371043838662e-14*(1.0 - cos(theta)**2)**5.5*(4.39768046580754e+16*cos(theta)**7 - 2.63860827948452e+16*cos(theta)**5 + 3.9978913325523e+15*cos(theta)**3 - 128964236533945.0*cos(theta))*sin(11*phi)

@torch.jit.script
def Yl18_m_minus_10(theta, phi):
    return 8.82471682796557e-13*(1.0 - cos(theta)**2)**5*(5.49710058225942e+15*cos(theta)**8 - 4.39768046580754e+15*cos(theta)**6 + 999472833138076.0*cos(theta)**4 - 64482118266972.7*cos(theta)**2 + 555880329887.695)*sin(10*phi)

@torch.jit.script
def Yl18_m_minus_9(theta, phi):
    return 1.40088036704182e-11*(1.0 - cos(theta)**2)**4.5*(610788953584380.0*cos(theta)**9 - 628240066543934.0*cos(theta)**7 + 199894566627615.0*cos(theta)**5 - 21494039422324.2*cos(theta)**3 + 555880329887.695*cos(theta))*sin(9*phi)

@torch.jit.script
def Yl18_m_minus_8(theta, phi):
    return 2.30188133218476e-10*(1.0 - cos(theta)**2)**4*(61078895358438.0*cos(theta)**10 - 78530008317991.7*cos(theta)**8 + 33315761104602.5*cos(theta)**6 - 5373509855581.05*cos(theta)**4 + 277940164943.848*cos(theta)**2 - 2058816036.62109)*sin(8*phi)

@torch.jit.script
def Yl18_m_minus_7(theta, phi):
    return 3.8928345622358e-9*(1.0 - cos(theta)**2)**3.5*(5552626850767.09*cos(theta)**11 - 8725556479776.86*cos(theta)**9 + 4759394443514.65*cos(theta)**7 - 1074701971116.21*cos(theta)**5 + 92646721647.9492*cos(theta)**3 - 2058816036.62109*cos(theta))*sin(7*phi)

@torch.jit.script
def Yl18_m_minus_6(theta, phi):
    return 6.74258724725256e-8*(1.0 - cos(theta)**2)**3*(462718904230.591*cos(theta)**12 - 872555647977.686*cos(theta)**10 + 594924305439.331*cos(theta)**8 - 179116995186.035*cos(theta)**6 + 23161680411.9873*cos(theta)**4 - 1029408018.31055*cos(theta)**2 + 6862720.12207031)*sin(6*phi)

@torch.jit.script
def Yl18_m_minus_5(theta, phi):
    return 1.19097836376173e-6*(1.0 - cos(theta)**2)**2.5*(35593761863.8916*cos(theta)**13 - 79323240725.2441*cos(theta)**11 + 66102700604.3701*cos(theta)**9 - 25588142169.4336*cos(theta)**7 + 4632336082.39746*cos(theta)**5 - 343136006.103516*cos(theta)**3 + 6862720.12207031*cos(theta))*sin(5*phi)

@torch.jit.script
def Yl18_m_minus_4(theta, phi):
    return 2.13713426594923e-5*(1.0 - cos(theta)**2)**2*(2542411561.70654*cos(theta)**14 - 6610270060.43701*cos(theta)**12 + 6610270060.43701*cos(theta)**10 - 3198517771.1792*cos(theta)**8 + 772056013.73291*cos(theta)**6 - 85784001.5258789*cos(theta)**4 + 3431360.06103516*cos(theta)**2 - 21312.7954101563)*sin(4*phi)

@torch.jit.script
def Yl18_m_minus_3(theta, phi):
    return 0.000388229719023305*(1.0 - cos(theta)**2)**1.5*(169494104.11377*cos(theta)**15 - 508482312.341309*cos(theta)**13 + 600933641.85791*cos(theta)**11 - 355390863.464355*cos(theta)**9 + 110293716.247559*cos(theta)**7 - 17156800.3051758*cos(theta)**5 + 1143786.68701172*cos(theta)**3 - 21312.7954101563*cos(theta))*sin(3*phi)

@torch.jit.script
def Yl18_m_minus_2(theta, phi):
    return 0.00711636829782292*(1.0 - cos(theta)**2)*(10593381.5071106*cos(theta)**16 - 36320165.1672363*cos(theta)**14 + 50077803.4881592*cos(theta)**12 - 35539086.3464355*cos(theta)**10 + 13786714.5309448*cos(theta)**8 - 2859466.7175293*cos(theta)**6 + 285946.67175293*cos(theta)**4 - 10656.3977050781*cos(theta)**2 + 63.4309387207031)*sin(2*phi)

@torch.jit.script
def Yl18_m_minus_1(theta, phi):
    return 0.131219347792496*(1.0 - cos(theta)**2)**0.5*(623140.088653564*cos(theta)**17 - 2421344.34448242*cos(theta)**15 + 3852138.7298584*cos(theta)**13 - 3230826.03149414*cos(theta)**11 + 1531857.17010498*cos(theta)**9 - 408495.245361328*cos(theta)**7 + 57189.3343505859*cos(theta)**5 - 3552.13256835938*cos(theta)**3 + 63.4309387207031*cos(theta))*sin(phi)

@torch.jit.script
def Yl18_m0(theta, phi):
    return 59403.1009679377*cos(theta)**18 - 259676.412802699*cos(theta)**16 + 472138.932368544*cos(theta)**14 - 461985.406941263*cos(theta)**12 + 262853.766018305*cos(theta)**10 - 87617.9220061016*cos(theta)**8 + 16355.345441139*cos(theta)**6 - 1523.7899479322*cos(theta)**4 + 54.4210695690072*cos(theta)**2 - 0.318251868824604

@torch.jit.script
def Yl18_m1(theta, phi):
    return 0.131219347792496*(1.0 - cos(theta)**2)**0.5*(623140.088653564*cos(theta)**17 - 2421344.34448242*cos(theta)**15 + 3852138.7298584*cos(theta)**13 - 3230826.03149414*cos(theta)**11 + 1531857.17010498*cos(theta)**9 - 408495.245361328*cos(theta)**7 + 57189.3343505859*cos(theta)**5 - 3552.13256835938*cos(theta)**3 + 63.4309387207031*cos(theta))*cos(phi)

@torch.jit.script
def Yl18_m2(theta, phi):
    return 0.00711636829782292*(1.0 - cos(theta)**2)*(10593381.5071106*cos(theta)**16 - 36320165.1672363*cos(theta)**14 + 50077803.4881592*cos(theta)**12 - 35539086.3464355*cos(theta)**10 + 13786714.5309448*cos(theta)**8 - 2859466.7175293*cos(theta)**6 + 285946.67175293*cos(theta)**4 - 10656.3977050781*cos(theta)**2 + 63.4309387207031)*cos(2*phi)

@torch.jit.script
def Yl18_m3(theta, phi):
    return 0.000388229719023305*(1.0 - cos(theta)**2)**1.5*(169494104.11377*cos(theta)**15 - 508482312.341309*cos(theta)**13 + 600933641.85791*cos(theta)**11 - 355390863.464355*cos(theta)**9 + 110293716.247559*cos(theta)**7 - 17156800.3051758*cos(theta)**5 + 1143786.68701172*cos(theta)**3 - 21312.7954101563*cos(theta))*cos(3*phi)

@torch.jit.script
def Yl18_m4(theta, phi):
    return 2.13713426594923e-5*(1.0 - cos(theta)**2)**2*(2542411561.70654*cos(theta)**14 - 6610270060.43701*cos(theta)**12 + 6610270060.43701*cos(theta)**10 - 3198517771.1792*cos(theta)**8 + 772056013.73291*cos(theta)**6 - 85784001.5258789*cos(theta)**4 + 3431360.06103516*cos(theta)**2 - 21312.7954101563)*cos(4*phi)

@torch.jit.script
def Yl18_m5(theta, phi):
    return 1.19097836376173e-6*(1.0 - cos(theta)**2)**2.5*(35593761863.8916*cos(theta)**13 - 79323240725.2441*cos(theta)**11 + 66102700604.3701*cos(theta)**9 - 25588142169.4336*cos(theta)**7 + 4632336082.39746*cos(theta)**5 - 343136006.103516*cos(theta)**3 + 6862720.12207031*cos(theta))*cos(5*phi)

@torch.jit.script
def Yl18_m6(theta, phi):
    return 6.74258724725256e-8*(1.0 - cos(theta)**2)**3*(462718904230.591*cos(theta)**12 - 872555647977.686*cos(theta)**10 + 594924305439.331*cos(theta)**8 - 179116995186.035*cos(theta)**6 + 23161680411.9873*cos(theta)**4 - 1029408018.31055*cos(theta)**2 + 6862720.12207031)*cos(6*phi)

@torch.jit.script
def Yl18_m7(theta, phi):
    return 3.8928345622358e-9*(1.0 - cos(theta)**2)**3.5*(5552626850767.09*cos(theta)**11 - 8725556479776.86*cos(theta)**9 + 4759394443514.65*cos(theta)**7 - 1074701971116.21*cos(theta)**5 + 92646721647.9492*cos(theta)**3 - 2058816036.62109*cos(theta))*cos(7*phi)

@torch.jit.script
def Yl18_m8(theta, phi):
    return 2.30188133218476e-10*(1.0 - cos(theta)**2)**4*(61078895358438.0*cos(theta)**10 - 78530008317991.7*cos(theta)**8 + 33315761104602.5*cos(theta)**6 - 5373509855581.05*cos(theta)**4 + 277940164943.848*cos(theta)**2 - 2058816036.62109)*cos(8*phi)

@torch.jit.script
def Yl18_m9(theta, phi):
    return 1.40088036704182e-11*(1.0 - cos(theta)**2)**4.5*(610788953584380.0*cos(theta)**9 - 628240066543934.0*cos(theta)**7 + 199894566627615.0*cos(theta)**5 - 21494039422324.2*cos(theta)**3 + 555880329887.695*cos(theta))*cos(9*phi)

@torch.jit.script
def Yl18_m10(theta, phi):
    return 8.82471682796557e-13*(1.0 - cos(theta)**2)**5*(5.49710058225942e+15*cos(theta)**8 - 4.39768046580754e+15*cos(theta)**6 + 999472833138076.0*cos(theta)**4 - 64482118266972.7*cos(theta)**2 + 555880329887.695)*cos(10*phi)

@torch.jit.script
def Yl18_m11(theta, phi):
    return 5.79371043838662e-14*(1.0 - cos(theta)**2)**5.5*(4.39768046580754e+16*cos(theta)**7 - 2.63860827948452e+16*cos(theta)**5 + 3.9978913325523e+15*cos(theta)**3 - 128964236533945.0*cos(theta))*cos(11*phi)

@torch.jit.script
def Yl18_m12(theta, phi):
    return 3.9980400343329e-15*(1.0 - cos(theta)**2)**6*(3.07837632606527e+17*cos(theta)**6 - 1.31930413974226e+17*cos(theta)**4 + 1.19936739976569e+16*cos(theta)**2 - 128964236533945.0)*cos(12*phi)

@torch.jit.script
def Yl18_m13(theta, phi):
    return 2.93150518387396e-16*(1.0 - cos(theta)**2)**6.5*(1.84702579563916e+18*cos(theta)**5 - 5.27721655896904e+17*cos(theta)**3 + 2.39873479953138e+16*cos(theta))*cos(13*phi)

@torch.jit.script
def Yl18_m14(theta, phi):
    return 2.31755833840811e-17*(1.0 - cos(theta)**2)**7*(9.23512897819582e+18*cos(theta)**4 - 1.58316496769071e+18*cos(theta)**2 + 2.39873479953138e+16)*cos(14*phi)

@torch.jit.script
def Yl18_m15(theta, phi):
    return 2.01717561545333e-18*(1.0 - cos(theta)**2)**7.5*(3.69405159127833e+19*cos(theta)**3 - 3.16632993538143e+18*cos(theta))*cos(15*phi)

@torch.jit.script
def Yl18_m16(theta, phi):
    return 1.99730147939357e-19*(1.0 - cos(theta)**2)**8*(1.1082154773835e+20*cos(theta)**2 - 3.16632993538143e+18)*cos(16*phi)

@torch.jit.script
def Yl18_m17(theta, phi):
    return 5.29113461206997*(1.0 - cos(theta)**2)**8.5*cos(17*phi)*cos(theta)

@torch.jit.script
def Yl18_m18(theta, phi):
    return 0.881855768678329*(1.0 - cos(theta)**2)**9*cos(18*phi)

@torch.jit.script
def Yl19_m_minus_19(theta, phi):
    return 0.893383784349949*(1.0 - cos(theta)**2)**9.5*sin(19*phi)

@torch.jit.script
def Yl19_m_minus_18(theta, phi):
    return 5.50718751027224*(1.0 - cos(theta)**2)**9*sin(18*phi)*cos(theta)

@torch.jit.script
def Yl19_m_minus_17(theta, phi):
    return 5.77683273022057e-21*(1.0 - cos(theta)**2)**8.5*(4.10039726631895e+21*cos(theta)**2 - 1.1082154773835e+20)*sin(17*phi)

@torch.jit.script
def Yl19_m_minus_16(theta, phi):
    return 6.00346067734132e-20*(1.0 - cos(theta)**2)**8*(1.36679908877298e+21*cos(theta)**3 - 1.1082154773835e+20*cos(theta))*sin(16*phi)

@torch.jit.script
def Yl19_m_minus_15(theta, phi):
    return 7.1033904683705e-19*(1.0 - cos(theta)**2)**7.5*(3.41699772193245e+20*cos(theta)**4 - 5.54107738691749e+19*cos(theta)**2 + 7.91582483845356e+17)*sin(15*phi)

@torch.jit.script
def Yl19_m_minus_14(theta, phi):
    return 9.26168804529891e-18*(1.0 - cos(theta)**2)**7*(6.83399544386491e+19*cos(theta)**5 - 1.84702579563916e+19*cos(theta)**3 + 7.91582483845356e+17*cos(theta))*sin(14*phi)

@torch.jit.script
def Yl19_m_minus_13(theta, phi):
    return 1.30323502710715e-16*(1.0 - cos(theta)**2)**6.5*(1.13899924064415e+19*cos(theta)**6 - 4.61756448909791e+18*cos(theta)**4 + 3.95791241922678e+17*cos(theta)**2 - 3.9978913325523e+15)*sin(13*phi)

@torch.jit.script
def Yl19_m_minus_12(theta, phi):
    return 1.9505035863512e-15*(1.0 - cos(theta)**2)**6*(1.62714177234879e+18*cos(theta)**7 - 9.23512897819582e+17*cos(theta)**5 + 1.31930413974226e+17*cos(theta)**3 - 3.9978913325523e+15*cos(theta))*sin(12*phi)

@torch.jit.script
def Yl19_m_minus_11(theta, phi):
    return 3.07165611944352e-14*(1.0 - cos(theta)**2)**5.5*(2.03392721543598e+17*cos(theta)**8 - 1.53918816303264e+17*cos(theta)**6 + 3.29826034935565e+16*cos(theta)**4 - 1.99894566627615e+15*cos(theta)**2 + 16120529566743.2)*sin(11*phi)

@torch.jit.script
def Yl19_m_minus_10(theta, phi):
    return 5.047246036554e-13*(1.0 - cos(theta)**2)**5*(2.25991912826221e+16*cos(theta)**9 - 2.19884023290377e+16*cos(theta)**7 + 6.5965206987113e+15*cos(theta)**5 - 666315222092051.0*cos(theta)**3 + 16120529566743.2*cos(theta))*sin(10*phi)

@torch.jit.script
def Yl19_m_minus_9(theta, phi):
    return 8.59515028403688e-12*(1.0 - cos(theta)**2)**4.5*(2.25991912826221e+15*cos(theta)**10 - 2.74855029112971e+15*cos(theta)**8 + 1.09942011645188e+15*cos(theta)**6 - 166578805523013.0*cos(theta)**4 + 8060264783371.58*cos(theta)**2 - 55588032988.7695)*sin(9*phi)

@torch.jit.script
def Yl19_m_minus_8(theta, phi):
    return 1.50844275293414e-10*(1.0 - cos(theta)**2)**4*(205447193478382.0*cos(theta)**11 - 305394476792190.0*cos(theta)**9 + 157060016635983.0*cos(theta)**7 - 33315761104602.5*cos(theta)**5 + 2686754927790.53*cos(theta)**3 - 55588032988.7695*cos(theta))*sin(8*phi)

@torch.jit.script
def Yl19_m_minus_7(theta, phi):
    return 2.71519695528145e-9*(1.0 - cos(theta)**2)**3.5*(17120599456531.9*cos(theta)**12 - 30539447679219.0*cos(theta)**10 + 19632502079497.9*cos(theta)**8 - 5552626850767.09*cos(theta)**6 + 671688731947.632*cos(theta)**4 - 27794016494.3848*cos(theta)**2 + 171568003.051758)*sin(7*phi)

@torch.jit.script
def Yl19_m_minus_6(theta, phi):
    return 4.99182886627511e-8*(1.0 - cos(theta)**2)**3*(1316969188963.99*cos(theta)**13 - 2776313425383.54*cos(theta)**11 + 2181389119944.21*cos(theta)**9 - 793232407252.441*cos(theta)**7 + 134337746389.526*cos(theta)**5 - 9264672164.79492*cos(theta)**3 + 171568003.051758*cos(theta))*sin(6*phi)

@torch.jit.script
def Yl19_m_minus_5(theta, phi):
    return 9.33885667550482e-7*(1.0 - cos(theta)**2)**2.5*(94069227783.1421*cos(theta)**14 - 231359452115.295*cos(theta)**12 + 218138911994.421*cos(theta)**10 - 99154050906.5552*cos(theta)**8 + 22389624398.2544*cos(theta)**6 - 2316168041.19873*cos(theta)**4 + 85784001.5258789*cos(theta)**2 - 490194.294433594)*sin(5*phi)

@torch.jit.script
def Yl19_m_minus_4(theta, phi):
    return 1.77192347018779e-5*(1.0 - cos(theta)**2)**2*(6271281852.20947*cos(theta)**15 - 17796880931.9458*cos(theta)**13 + 19830810181.311*cos(theta)**11 - 11017116767.395*cos(theta)**9 + 3198517771.1792*cos(theta)**7 - 463233608.239746*cos(theta)**5 + 28594667.175293*cos(theta)**3 - 490194.294433594*cos(theta))*sin(4*phi)

@torch.jit.script
def Yl19_m_minus_3(theta, phi):
    return 0.000339913857408971*(1.0 - cos(theta)**2)**1.5*(391955115.763092*cos(theta)**16 - 1271205780.85327*cos(theta)**14 + 1652567515.10925*cos(theta)**12 - 1101711676.7395*cos(theta)**10 + 399814721.3974*cos(theta)**8 - 77205601.373291*cos(theta)**6 + 7148666.79382324*cos(theta)**4 - 245097.147216797*cos(theta)**2 + 1332.04971313477)*sin(3*phi)

@torch.jit.script
def Yl19_m_minus_2(theta, phi):
    return 0.00657362114755131*(1.0 - cos(theta)**2)*(23056183.2801819*cos(theta)**17 - 84747052.0568848*cos(theta)**15 + 127120578.085327*cos(theta)**13 - 100155606.976318*cos(theta)**11 + 44423857.9330444*cos(theta)**9 - 11029371.6247559*cos(theta)**7 + 1429733.35876465*cos(theta)**5 - 81699.0490722656*cos(theta)**3 + 1332.04971313477*cos(theta))*sin(2*phi)

@torch.jit.script
def Yl19_m_minus_1(theta, phi):
    return 0.127805802320551*(1.0 - cos(theta)**2)**0.5*(1280899.07112122*cos(theta)**18 - 5296690.7535553*cos(theta)**16 + 9080041.29180908*cos(theta)**14 - 8346300.58135986*cos(theta)**12 + 4442385.79330444*cos(theta)**10 - 1378671.45309448*cos(theta)**8 + 238288.893127441*cos(theta)**6 - 20424.7622680664*cos(theta)**4 + 666.024856567383*cos(theta)**2 - 3.52394104003906)*sin(phi)

@torch.jit.script
def Yl19_m0(theta, phi):
    return 118765.056929642*cos(theta)**19 - 548887.154999156*cos(theta)**17 + 1066409.32971265*cos(theta)**15 - 1131040.19818008*cos(theta)**13 + 711460.769822953*cos(theta)**11 - 269864.429932844*cos(theta)**9 + 59969.8733184099*cos(theta)**7 - 7196.38479820918*cos(theta)**5 + 391.10786946789*cos(theta)**3 - 6.20806142012525*cos(theta)

@torch.jit.script
def Yl19_m1(theta, phi):
    return 0.127805802320551*(1.0 - cos(theta)**2)**0.5*(1280899.07112122*cos(theta)**18 - 5296690.7535553*cos(theta)**16 + 9080041.29180908*cos(theta)**14 - 8346300.58135986*cos(theta)**12 + 4442385.79330444*cos(theta)**10 - 1378671.45309448*cos(theta)**8 + 238288.893127441*cos(theta)**6 - 20424.7622680664*cos(theta)**4 + 666.024856567383*cos(theta)**2 - 3.52394104003906)*cos(phi)

@torch.jit.script
def Yl19_m2(theta, phi):
    return 0.00657362114755131*(1.0 - cos(theta)**2)*(23056183.2801819*cos(theta)**17 - 84747052.0568848*cos(theta)**15 + 127120578.085327*cos(theta)**13 - 100155606.976318*cos(theta)**11 + 44423857.9330444*cos(theta)**9 - 11029371.6247559*cos(theta)**7 + 1429733.35876465*cos(theta)**5 - 81699.0490722656*cos(theta)**3 + 1332.04971313477*cos(theta))*cos(2*phi)

@torch.jit.script
def Yl19_m3(theta, phi):
    return 0.000339913857408971*(1.0 - cos(theta)**2)**1.5*(391955115.763092*cos(theta)**16 - 1271205780.85327*cos(theta)**14 + 1652567515.10925*cos(theta)**12 - 1101711676.7395*cos(theta)**10 + 399814721.3974*cos(theta)**8 - 77205601.373291*cos(theta)**6 + 7148666.79382324*cos(theta)**4 - 245097.147216797*cos(theta)**2 + 1332.04971313477)*cos(3*phi)

@torch.jit.script
def Yl19_m4(theta, phi):
    return 1.77192347018779e-5*(1.0 - cos(theta)**2)**2*(6271281852.20947*cos(theta)**15 - 17796880931.9458*cos(theta)**13 + 19830810181.311*cos(theta)**11 - 11017116767.395*cos(theta)**9 + 3198517771.1792*cos(theta)**7 - 463233608.239746*cos(theta)**5 + 28594667.175293*cos(theta)**3 - 490194.294433594*cos(theta))*cos(4*phi)

@torch.jit.script
def Yl19_m5(theta, phi):
    return 9.33885667550482e-7*(1.0 - cos(theta)**2)**2.5*(94069227783.1421*cos(theta)**14 - 231359452115.295*cos(theta)**12 + 218138911994.421*cos(theta)**10 - 99154050906.5552*cos(theta)**8 + 22389624398.2544*cos(theta)**6 - 2316168041.19873*cos(theta)**4 + 85784001.5258789*cos(theta)**2 - 490194.294433594)*cos(5*phi)

@torch.jit.script
def Yl19_m6(theta, phi):
    return 4.99182886627511e-8*(1.0 - cos(theta)**2)**3*(1316969188963.99*cos(theta)**13 - 2776313425383.54*cos(theta)**11 + 2181389119944.21*cos(theta)**9 - 793232407252.441*cos(theta)**7 + 134337746389.526*cos(theta)**5 - 9264672164.79492*cos(theta)**3 + 171568003.051758*cos(theta))*cos(6*phi)

@torch.jit.script
def Yl19_m7(theta, phi):
    return 2.71519695528145e-9*(1.0 - cos(theta)**2)**3.5*(17120599456531.9*cos(theta)**12 - 30539447679219.0*cos(theta)**10 + 19632502079497.9*cos(theta)**8 - 5552626850767.09*cos(theta)**6 + 671688731947.632*cos(theta)**4 - 27794016494.3848*cos(theta)**2 + 171568003.051758)*cos(7*phi)

@torch.jit.script
def Yl19_m8(theta, phi):
    return 1.50844275293414e-10*(1.0 - cos(theta)**2)**4*(205447193478382.0*cos(theta)**11 - 305394476792190.0*cos(theta)**9 + 157060016635983.0*cos(theta)**7 - 33315761104602.5*cos(theta)**5 + 2686754927790.53*cos(theta)**3 - 55588032988.7695*cos(theta))*cos(8*phi)

@torch.jit.script
def Yl19_m9(theta, phi):
    return 8.59515028403688e-12*(1.0 - cos(theta)**2)**4.5*(2.25991912826221e+15*cos(theta)**10 - 2.74855029112971e+15*cos(theta)**8 + 1.09942011645188e+15*cos(theta)**6 - 166578805523013.0*cos(theta)**4 + 8060264783371.58*cos(theta)**2 - 55588032988.7695)*cos(9*phi)

@torch.jit.script
def Yl19_m10(theta, phi):
    return 5.047246036554e-13*(1.0 - cos(theta)**2)**5*(2.25991912826221e+16*cos(theta)**9 - 2.19884023290377e+16*cos(theta)**7 + 6.5965206987113e+15*cos(theta)**5 - 666315222092051.0*cos(theta)**3 + 16120529566743.2*cos(theta))*cos(10*phi)

@torch.jit.script
def Yl19_m11(theta, phi):
    return 3.07165611944352e-14*(1.0 - cos(theta)**2)**5.5*(2.03392721543598e+17*cos(theta)**8 - 1.53918816303264e+17*cos(theta)**6 + 3.29826034935565e+16*cos(theta)**4 - 1.99894566627615e+15*cos(theta)**2 + 16120529566743.2)*cos(11*phi)

@torch.jit.script
def Yl19_m12(theta, phi):
    return 1.9505035863512e-15*(1.0 - cos(theta)**2)**6*(1.62714177234879e+18*cos(theta)**7 - 9.23512897819582e+17*cos(theta)**5 + 1.31930413974226e+17*cos(theta)**3 - 3.9978913325523e+15*cos(theta))*cos(12*phi)

@torch.jit.script
def Yl19_m13(theta, phi):
    return 1.30323502710715e-16*(1.0 - cos(theta)**2)**6.5*(1.13899924064415e+19*cos(theta)**6 - 4.61756448909791e+18*cos(theta)**4 + 3.95791241922678e+17*cos(theta)**2 - 3.9978913325523e+15)*cos(13*phi)

@torch.jit.script
def Yl19_m14(theta, phi):
    return 9.26168804529891e-18*(1.0 - cos(theta)**2)**7*(6.83399544386491e+19*cos(theta)**5 - 1.84702579563916e+19*cos(theta)**3 + 7.91582483845356e+17*cos(theta))*cos(14*phi)

@torch.jit.script
def Yl19_m15(theta, phi):
    return 7.1033904683705e-19*(1.0 - cos(theta)**2)**7.5*(3.41699772193245e+20*cos(theta)**4 - 5.54107738691749e+19*cos(theta)**2 + 7.91582483845356e+17)*cos(15*phi)

@torch.jit.script
def Yl19_m16(theta, phi):
    return 6.00346067734132e-20*(1.0 - cos(theta)**2)**8*(1.36679908877298e+21*cos(theta)**3 - 1.1082154773835e+20*cos(theta))*cos(16*phi)

@torch.jit.script
def Yl19_m17(theta, phi):
    return 5.77683273022057e-21*(1.0 - cos(theta)**2)**8.5*(4.10039726631895e+21*cos(theta)**2 - 1.1082154773835e+20)*cos(17*phi)

@torch.jit.script
def Yl19_m18(theta, phi):
    return 5.50718751027224*(1.0 - cos(theta)**2)**9*cos(18*phi)*cos(theta)

@torch.jit.script
def Yl19_m19(theta, phi):
    return 0.893383784349949*(1.0 - cos(theta)**2)**9.5*cos(19*phi)
