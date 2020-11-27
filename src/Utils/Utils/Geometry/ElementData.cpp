/**
 * @file ElementData.cpp
 * @copyright This code is licensed under the 3-clause BSD license.\n
 *            Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.\n
 *            See LICENSE.txt for details.
 */
#include "Utils/Geometry/ElementData.h"
#include "Utils/Geometry/ElementInfo.h"

namespace Scine {
namespace Utils {
namespace Constants {

std::unique_ptr<ElementDataSingleton> ElementDataSingleton::d_instance = nullptr;

ElementDataSingleton::ElementDataSingleton() {
  init_data();
}

const ElementDataSingleton::ElementData& ElementDataSingleton::operator[](const ElementType& type) const {
  return d_container.at(type);
}

const ElementDataSingleton::ElementData& ElementDataSingleton::operator[](const std::string& symbol) const {
  auto type = ElementInfo::elementTypeForSymbol(symbol);
  return d_container.at(type);
}

const ElementDataSingleton& ElementDataSingleton::instance() {
  if (!d_instance) {
#pragma omp critical(ElementDataSingletonInstance)
    { d_instance = std::make_unique<ElementDataSingleton>(ElementDataSingleton()); }
  }
  return *d_instance;
}

void ElementDataSingleton::init_data() {
  // clang-format off
  d_container.emplace(ElementType::H,  ElementData("H",    1, 1.0079, 32 , 109));
  d_container.emplace(ElementType::He, ElementData("He",   2, 4.0026, 37 , 140));
  d_container.emplace(ElementType::Li, ElementData("Li",   3,  6.941, 130, 182));
  d_container.emplace(ElementType::Be, ElementData("Be",   4, 9.0122, 99 , 153));
  d_container.emplace(ElementType::B,  ElementData("B",    5, 10.811, 84 , 192));
  d_container.emplace(ElementType::C,  ElementData("C",    6, 12.011, 75 , 170));
  d_container.emplace(ElementType::N,  ElementData("N",    7, 14.007, 71 , 155));
  d_container.emplace(ElementType::O,  ElementData("O",    8, 15.999, 64 , 152));
  d_container.emplace(ElementType::F,  ElementData("F",    9, 18.988, 60 , 147));
  d_container.emplace(ElementType::Ne, ElementData("Ne",  10, 20.180, 62 , 154));
  d_container.emplace(ElementType::Na, ElementData("Na",  11, 22.990, 160, 227));
  d_container.emplace(ElementType::Mg, ElementData("Mg",  12, 24.305, 140, 173));
  d_container.emplace(ElementType::Al, ElementData("Al",  13, 26.982, 124, 184));
  d_container.emplace(ElementType::Si, ElementData("Si",  14, 28.086, 114, 210));
  d_container.emplace(ElementType::P,  ElementData("P",   15, 30.974, 109, 180));
  d_container.emplace(ElementType::S,  ElementData("S",   16, 32.065, 104, 180));
  d_container.emplace(ElementType::Cl, ElementData("Cl",  17, 35.453, 100, 175));
  d_container.emplace(ElementType::Ar, ElementData("Ar",  18, 39.948, 101, 188));
  d_container.emplace(ElementType::K,  ElementData("K",   19, 39.098, 200, 275));
  d_container.emplace(ElementType::Ca, ElementData("Ca",  20, 40.078, 174, 231));
  d_container.emplace(ElementType::Sc, ElementData("Sc",  21, 44.956, 159, 215));
  d_container.emplace(ElementType::Ti, ElementData("Ti",  22, 47.867, 148, 211));
  d_container.emplace(ElementType::V,  ElementData("V",   23, 50.942, 144, 207));
  d_container.emplace(ElementType::Cr, ElementData("Cr",  24, 51.996, 130, 206));
  d_container.emplace(ElementType::Mn, ElementData("Mn",  25, 54.938, 129, 205));
  d_container.emplace(ElementType::Fe, ElementData("Fe",  26, 55.938, 124, 204));
  d_container.emplace(ElementType::Co, ElementData("Co",  27, 58.933, 118, 200));
  d_container.emplace(ElementType::Ni, ElementData("Ni",  28, 58.693, 117, 197));
  d_container.emplace(ElementType::Cu, ElementData("Cu",  29, 63.546, 122, 196));
  d_container.emplace(ElementType::Zn, ElementData("Zn",  30,  65.38, 120, 201));
  d_container.emplace(ElementType::Ga, ElementData("Ga",  31, 69.723, 123, 187));
  d_container.emplace(ElementType::Ge, ElementData("Ge",  32,  72.64, 120, 211));
  d_container.emplace(ElementType::As, ElementData("As",  33, 74.922, 120, 185));
  d_container.emplace(ElementType::Se, ElementData("Se",  34,  78.96, 118, 190));
  d_container.emplace(ElementType::Br, ElementData("Br",  35, 79.904, 117, 185));
  d_container.emplace(ElementType::Kr, ElementData("Kr",  36, 83.798, 116, 202));
  d_container.emplace(ElementType::Rb, ElementData("Rb",  37, 83.468, 215, 303));
  d_container.emplace(ElementType::Sr, ElementData("Sr",  38,  87.62, 190, 249));
  d_container.emplace(ElementType::Y,  ElementData("Y",   39, 88.906, 176, 232));
  d_container.emplace(ElementType::Zr, ElementData("Zr",  40, 91.224, 164, 223));
  d_container.emplace(ElementType::Nb, ElementData("Nb",  41, 92.906, 156, 218));
  d_container.emplace(ElementType::Mo, ElementData("Mo",  42,  95.96, 146, 217));
  d_container.emplace(ElementType::Tc, ElementData("Tc",  43,  98.91, 138, 216));
  d_container.emplace(ElementType::Ru, ElementData("Ru",  44, 101.07, 136, 213));
  d_container.emplace(ElementType::Rh, ElementData("Rh",  45, 102.91, 134, 210));
  d_container.emplace(ElementType::Pd, ElementData("Pd",  46, 106.42, 130, 210));
  d_container.emplace(ElementType::Ag, ElementData("Ag",  47, 107.87, 136, 211));
  d_container.emplace(ElementType::Cd, ElementData("Cd",  48, 112.41, 140, 218));
  d_container.emplace(ElementType::In, ElementData("In",  49, 114.82, 142, 193));
  d_container.emplace(ElementType::Sn, ElementData("Sn",  50, 118.71, 140, 217));
  d_container.emplace(ElementType::Sb, ElementData("Sb",  51, 121.76, 140, 206));
  d_container.emplace(ElementType::Te, ElementData("Te",  52, 127.60, 137, 206));
  d_container.emplace(ElementType::I,  ElementData("I",   53, 126.90, 136, 198));
  d_container.emplace(ElementType::Xe, ElementData("Xe",  54, 131.29, 136, 216));
  d_container.emplace(ElementType::Cs, ElementData("Cs",  55, 132.91, 238, 343));
  d_container.emplace(ElementType::Ba, ElementData("Ba",  56, 137.33, 206, 268));
  d_container.emplace(ElementType::La, ElementData("La",  57, 138.91, 194, 243));
  d_container.emplace(ElementType::Ce, ElementData("Ce",  58, 140.12, 184, 242));
  d_container.emplace(ElementType::Pr, ElementData("Pr",  59, 140.91, 190, 240));
  d_container.emplace(ElementType::Nd, ElementData("Nd",  60, 144.24, 188, 239));
  d_container.emplace(ElementType::Pm, ElementData("Pm",  61, 146.90, 186, 238));
  d_container.emplace(ElementType::Sm, ElementData("Sm",  62, 150.36, 185, 236));
  d_container.emplace(ElementType::Eu, ElementData("Eu",  63, 151.96, 183, 235));
  d_container.emplace(ElementType::Gd, ElementData("Gd",  64, 157.25, 182, 234));
  d_container.emplace(ElementType::Tb, ElementData("Tb",  65, 158.93, 181, 233));
  d_container.emplace(ElementType::Dy, ElementData("Dy",  66, 162.50, 180, 231));
  d_container.emplace(ElementType::Ho, ElementData("Ho",  67, 164.93, 179, 230));
  d_container.emplace(ElementType::Er, ElementData("Er",  68, 167.26, 177, 229));
  d_container.emplace(ElementType::Tm, ElementData("Tm",  69, 168.93, 177, 227));
  d_container.emplace(ElementType::Yb, ElementData("Yb",  70, 173.05, 178, 226));
  d_container.emplace(ElementType::Lu, ElementData("Lu",  71, 174.97, 174, 224));
  d_container.emplace(ElementType::Hf, ElementData("Hf",  72, 178.49, 164, 223));
  d_container.emplace(ElementType::Ta, ElementData("Ta",  73, 180.95, 158, 222));
  d_container.emplace(ElementType::W,  ElementData("W",   74, 183.84, 150, 218));
  d_container.emplace(ElementType::Re, ElementData("Re",  75, 186.21, 141, 216));
  d_container.emplace(ElementType::Os, ElementData("Os",  76, 190.23, 136, 216));
  d_container.emplace(ElementType::Ir, ElementData("Ir",  77, 192.22, 132, 213));
  d_container.emplace(ElementType::Pt, ElementData("Pt",  78, 195.08, 130, 213));
  d_container.emplace(ElementType::Au, ElementData("Au",  79, 196.97, 130, 214));
  d_container.emplace(ElementType::Hg, ElementData("Hg",  80, 200.59, 132, 223));
  d_container.emplace(ElementType::Tl, ElementData("Tl",  81, 204.38, 144, 196));
  d_container.emplace(ElementType::Pb, ElementData("Pb",  82,  207.2, 145, 202));
  d_container.emplace(ElementType::Bi, ElementData("Bi",  83, 208.98, 150, 207));
  d_container.emplace(ElementType::Po, ElementData("Po",  84, 209.98, 142, 197));
  d_container.emplace(ElementType::At, ElementData("At",  85,    210, 148, 202));
  d_container.emplace(ElementType::Rn, ElementData("Rn",  86,    222, 146, 220));
  d_container.emplace(ElementType::Fr, ElementData("Fr",  87,    223, 242, 348));
  d_container.emplace(ElementType::Ra, ElementData("Ra",  88, 226.03, 211, 283));
  d_container.emplace(ElementType::Ac, ElementData("Ac",  89,    227, 201, 247));
  d_container.emplace(ElementType::Th, ElementData("Th",  90, 232.04, 190, 245));
  d_container.emplace(ElementType::Pa, ElementData("Pa",  91, 231.04, 184, 243));
  d_container.emplace(ElementType::U,  ElementData("U",   92, 238.03, 183, 241));
  d_container.emplace(ElementType::Np, ElementData("Np",  93, 237.05, 180, 239));
  d_container.emplace(ElementType::Pu, ElementData("Pu",  94, 244.10, 180, 243));
  d_container.emplace(ElementType::Am, ElementData("Am",  95, 243.10, 173, 244));
  d_container.emplace(ElementType::Cm, ElementData("Cm",  96, 247.10, 168, 245));
  d_container.emplace(ElementType::Bk, ElementData("Bk",  97, 247.10, 168, 244));
  d_container.emplace(ElementType::Cf, ElementData("Cf",  98, 251.10, 168, 245));
  d_container.emplace(ElementType::Es, ElementData("Es",  99, 254.10, 165, 245));
  d_container.emplace(ElementType::Fm, ElementData("Fm", 100, 257.10, 167, 245));
  d_container.emplace(ElementType::Md, ElementData("Md", 101,    258, 173, 246));
  d_container.emplace(ElementType::No, ElementData("No", 102,    259, 176, 246));
  d_container.emplace(ElementType::Lr, ElementData("Lr", 103,    262, 161, 246));
  d_container.emplace(ElementType::Rf, ElementData("Rf", 104,    261, 157     ));
  d_container.emplace(ElementType::Db, ElementData("Db", 105,    262, 149     ));
  d_container.emplace(ElementType::Sg, ElementData("Sg", 106,    266, 143     ));
  d_container.emplace(ElementType::Bh, ElementData("Bh", 107,    264, 141     ));
  d_container.emplace(ElementType::Hs, ElementData("Hs", 108,    277, 134     ));
  d_container.emplace(ElementType::Mt, ElementData("Mt", 109,    268, 129     ));
  d_container.emplace(ElementType::Ds, ElementData("Ds", 110,    281, 128     ));
  d_container.emplace(ElementType::Rg, ElementData("Rg", 111,    280, 121     ));
  d_container.emplace(ElementType::Cn, ElementData("Cn", 112,    285, 122     ));
  // clang-format on
}

} /* namespace Constants */
} /* namespace Utils */
} /* namespace Scine */
