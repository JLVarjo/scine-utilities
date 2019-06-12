import pytest
import scine_utils_os as scine
import os

def test_IO():
  # Write a sample XYZ file
  sample_XYZ = """50			
			
Fe	-4.492570	-0.096320	0.847820
Fe	-2.054660	-0.157130	-1.789610
C	-1.813660	-1.875500	-1.459220
C	-4.969040	2.999280	0.925470
C	-6.249510	-0.257510	0.804750
C	-4.408740	0.041750	2.840480
C	-4.294330	-1.844000	1.009110
C	-2.107250	-0.591260	-3.740720
C	-0.313290	0.130170	-1.777170
C	-4.142080	1.410610	-2.161450
S	-4.408720	-0.238460	-1.600010
S	-2.183280	0.352810	0.608390
O	-1.664640	-3.009190	-1.274240
O	-4.155250	-2.985570	1.147960
O	0.835680	0.261970	-1.698400
O	-7.392040	-0.434060	0.719920
O	-3.105390	-0.364160	-4.402920
C	-0.869780	-1.168610	-4.413420
H	-0.133600	-0.361770	-4.517270
H	-1.116210	-1.559320	-5.405540
H	-0.413360	-1.946010	-3.794520
O	-3.501840	0.645880	3.386530
C	-5.517750	-0.577970	3.678110
H	-5.758910	-1.583080	3.321080
H	-6.419720	0.034770	3.556400
H	-5.237840	-0.599780	4.735860
N	-4.211750	1.881960	0.818590
C	-2.865400	1.974350	0.714200
C	-2.193970	3.191310	0.700110
C	-2.961220	4.348520	0.792630
C	-4.349030	4.249390	0.905800
C	-6.451570	2.842560	1.058480
H	-6.694570	2.226970	1.931260
H	-6.867270	2.346530	0.175060
H	-6.933400	3.815360	1.169820
H	-1.113360	3.219300	0.616820
H	-2.483250	5.323370	0.778410
H	-4.961470	5.141290	0.982050
N	-2.811460	1.626370	-2.282410
C	-2.348280	2.822830	-2.714460
C	-3.256430	3.839770	-3.013160
C	-4.629240	3.622250	-2.883990
C	-5.091920	2.382130	-2.454860
C	-0.868790	3.003810	-2.851970
H	-0.636440	4.001570	-3.228380
H	-0.455030	2.261230	-3.542570
H	-0.373260	2.871710	-1.884410
H	-2.879290	4.799710	-3.348950
H	-5.331970	4.415900	-3.118460
H	-6.148750	2.165060	-2.346420"""

  sample_MOL = """ 
 OpenBabel02081817133D

 50 54  0  0  0  0  0  0  0  0999 V2000
   -4.4926   -0.0963    0.8478 Fe  0  0  0  0  0  0  0  0  0  0  0  0
   -2.0547   -0.1571   -1.7896 Fe  0  0  0  0  0  0  0  0  0  0  0  0
   -1.8137   -1.8755   -1.4592 C   0  0  0  0  0  0  0  0  0  0  0  0
   -4.9690    2.9993    0.9255 C   0  0  0  0  0  0  0  0  0  0  0  0
   -6.2495   -0.2575    0.8047 C   0  0  0  0  0  0  0  0  0  0  0  0
   -4.4087    0.0418    2.8405 C   0  0  0  0  0  0  0  0  0  0  0  0
   -4.2943   -1.8440    1.0091 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.1073   -0.5913   -3.7407 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3133    0.1302   -1.7772 C   0  0  0  0  0  0  0  0  0  0  0  0
   -4.1421    1.4106   -2.1614 C   0  0  0  0  0  0  0  0  0  0  0  0
   -4.4087   -0.2385   -1.6000 S   0  0  2  0  0  0  0  0  0  0  0  0
   -2.1833    0.3528    0.6084 S   0  0  1  0  0  0  0  0  0  0  0  0
   -1.6646   -3.0092   -1.2742 O   0  0  0  0  0  0  0  0  0  0  0  0
   -4.1552   -2.9856    1.1480 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.8357    0.2620   -1.6984 O   0  0  0  0  0  0  0  0  0  0  0  0
   -7.3920   -0.4341    0.7199 O   0  0  0  0  0  0  0  0  0  0  0  0
   -3.1054   -0.3642   -4.4029 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8698   -1.1686   -4.4134 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.1336   -0.3618   -4.5173 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1162   -1.5593   -5.4055 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.4134   -1.9460   -3.7945 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.5018    0.6459    3.3865 O   0  0  0  0  0  0  0  0  0  0  0  0
   -5.5178   -0.5780    3.6781 C   0  0  0  0  0  0  0  0  0  0  0  0
   -5.7589   -1.5831    3.3211 H   0  0  0  0  0  0  0  0  0  0  0  0
   -6.4197    0.0348    3.5564 H   0  0  0  0  0  0  0  0  0  0  0  0
   -5.2378   -0.5998    4.7359 H   0  0  0  0  0  0  0  0  0  0  0  0
   -4.2118    1.8820    0.8186 N   0  0  0  0  0  0  0  0  0  0  0  0
   -2.8654    1.9744    0.7142 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.1940    3.1913    0.7001 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.9612    4.3485    0.7926 C   0  0  0  0  0  0  0  0  0  0  0  0
   -4.3490    4.2494    0.9058 C   0  0  0  0  0  0  0  0  0  0  0  0
   -6.4516    2.8426    1.0585 C   0  0  0  0  0  0  0  0  0  0  0  0
   -6.6946    2.2270    1.9313 H   0  0  0  0  0  0  0  0  0  0  0  0
   -6.8673    2.3465    0.1751 H   0  0  0  0  0  0  0  0  0  0  0  0
   -6.9334    3.8154    1.1698 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1134    3.2193    0.6168 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.4832    5.3234    0.7784 H   0  0  0  0  0  0  0  0  0  0  0  0
   -4.9615    5.1413    0.9820 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.8115    1.6264   -2.2824 N   0  0  0  0  0  0  0  0  0  0  0  0
   -2.3483    2.8228   -2.7145 C   0  0  0  0  0  0  0  0  0  0  0  0
   -3.2564    3.8398   -3.0132 C   0  0  0  0  0  0  0  0  0  0  0  0
   -4.6292    3.6223   -2.8840 C   0  0  0  0  0  0  0  0  0  0  0  0
   -5.0919    2.3821   -2.4549 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8688    3.0038   -2.8520 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6364    4.0016   -3.2284 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.4550    2.2612   -3.5426 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3733    2.8717   -1.8844 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.8793    4.7997   -3.3489 H   0  0  0  0  0  0  0  0  0  0  0  0
   -5.3320    4.4159   -3.1185 H   0  0  0  0  0  0  0  0  0  0  0  0
   -6.1487    2.1651   -2.3464 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  7  1  0  0  0  0
  1  6  1  0  0  0  0
  2  9  1  0  0  0  0
  2 11  1  0  0  0  0
  2  3  1  0  0  0  0
  3 13  1  0  0  0  0
  4 32  1  0  0  0  0
  5  1  1  0  0  0  0
  6 22  2  0  0  0  0
  6 23  1  0  0  0  0
  7 14  1  0  0  0  0
  8  2  1  0  0  0  0
  9 15  1  0  0  0  0
 11 10  1  6  0  0  0
 11  1  1  0  0  0  0
 12  2  1  6  0  0  0
 12 28  1  0  0  0  0
 12  1  1  0  0  0  0
 16  5  1  0  0  0  0
 17  8  2  0  0  0  0
 18 21  1  0  0  0  0
 18  8  1  0  0  0  0
 19 18  1  0  0  0  0
 20 18  1  0  0  0  0
 23 26  1  0  0  0  0
 24 23  1  0  0  0  0
 25 23  1  0  0  0  0
 27  1  1  0  0  0  0
 27  4  1  0  0  0  0
 28 27  1  0  0  0  0
 29 28  2  0  0  0  0
 29 30  1  0  0  0  0
 30 31  1  0  0  0  0
 31  4  2  0  0  0  0
 31 38  1  0  0  0  0
 32 35  1  0  0  0  0
 32 33  1  0  0  0  0
 34 32  1  0  0  0  0
 36 29  1  0  0  0  0
 37 30  1  0  0  0  0
 39 10  1  0  0  0  0
 39  2  1  0  0  0  0
 40 39  1  0  0  0  0
 41 42  1  0  0  0  0
 41 40  2  0  0  0  0
 42 43  1  0  0  0  0
 43 50  1  0  0  0  0
 43 10  2  0  0  0  0
 44 40  1  0  0  0  0
 44 47  1  0  0  0  0
 45 44  1  0  0  0  0
 46 44  1  0  0  0  0
 48 41  1  0  0  0  0
 49 42  1  0  0  0  0
M  END"""

  # Test reading an XYZ file
  with open("sample.xyz", "w") as xyz_file:
    xyz_file.write(sample_XYZ)

  (xyz_atoms, xyz_BO) = scine.IO.read("sample.xyz")
  assert xyz_BO.empty()
  assert xyz_atoms.size() == 50
  assert xyz_atoms.elements[0] == scine.ElementType.Fe

  # Write the read atoms into a file and re-read
  scine.IO.write("out.xyz", xyz_atoms)
  (reread_xyz_atoms, reread_xyz_BO) = scine.IO.read("out.xyz")
  assert xyz_atoms == reread_xyz_atoms

  # Clean up
  os.remove("sample.xyz")
  os.remove("out.xyz")

  # Test reading a MOL file
  with open("sample.mol", "w") as mol_file:
    mol_file.write(sample_MOL)

  (mol_atoms, mol_BO) = scine.IO.read("sample.mol")
  assert mol_atoms.size() == 50
  assert mol_atoms.elements[0] == scine.ElementType.Fe
  assert mol_BO.get_system_size() == 50

  # Write the read info into a file and re-read
  scine.IO.write_topology("out.mol", mol_atoms, mol_BO)
  (reread_mol_atoms, reread_mol_BO) = scine.IO.read("out.mol")
  assert mol_atoms == reread_mol_atoms
  assert mol_BO == reread_mol_BO

  # Clean up
  os.remove("sample.mol")
  os.remove("out.mol")