/**
 * @file
 * @copyright This code is licensed under the 3-clause BSD license.\n
 *            Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.\n
 *            See LICENSE.txt for details.
 */
#include <Utils/Geometry/ElementInfo.h>
#include <Utils/Geometry/ElementTypes.h>
#include <Utils/Pybind.h>

using namespace Scine::Utils;

void init_element_type(pybind11::module& m) {
  pybind11::enum_<ElementType> element_type(m, "ElementType", pybind11::arithmetic(),
                                            R"delim(
      Enum to represent element types including isotopes

      >>> h = ElementType.H # Represents isotopic mixture
      >>> h1 = ElementType.H1 # Represents only H with A = 1
      >>> d = ElementType.D # Represents only H with A = 2 (Deuterium)
    )delim");

  element_type.value("none", ElementType::none)
      .value("H", ElementType::H)
      .value("He", ElementType::He)
      .value("Li", ElementType::Li)
      .value("Be", ElementType::Be)
      .value("B", ElementType::B)
      .value("C", ElementType::C)
      .value("N", ElementType::N)
      .value("O", ElementType::O)
      .value("F", ElementType::F)
      .value("Ne", ElementType::Ne)
      .value("Na", ElementType::Na)
      .value("Mg", ElementType::Mg)
      .value("Al", ElementType::Al)
      .value("Si", ElementType::Si)
      .value("P", ElementType::P)
      .value("S", ElementType::S)
      .value("Cl", ElementType::Cl)
      .value("Ar", ElementType::Ar)
      .value("K", ElementType::K)
      .value("Ca", ElementType::Ca)
      .value("Sc", ElementType::Sc)
      .value("Ti", ElementType::Ti)
      .value("V", ElementType::V)
      .value("Cr", ElementType::Cr)
      .value("Mn", ElementType::Mn)
      .value("Fe", ElementType::Fe)
      .value("Co", ElementType::Co)
      .value("Ni", ElementType::Ni)
      .value("Cu", ElementType::Cu)
      .value("Zn", ElementType::Zn)
      .value("Ga", ElementType::Ga)
      .value("Ge", ElementType::Ge)
      .value("As", ElementType::As)
      .value("Se", ElementType::Se)
      .value("Br", ElementType::Br)
      .value("Kr", ElementType::Kr)
      .value("Rb", ElementType::Rb)
      .value("Sr", ElementType::Sr)
      .value("Y", ElementType::Y)
      .value("Zr", ElementType::Zr)
      .value("Nb", ElementType::Nb)
      .value("Mo", ElementType::Mo)
      .value("Tc", ElementType::Tc)
      .value("Ru", ElementType::Ru)
      .value("Rh", ElementType::Rh)
      .value("Pd", ElementType::Pd)
      .value("Ag", ElementType::Ag)
      .value("Cd", ElementType::Cd)
      .value("In", ElementType::In)
      .value("Sn", ElementType::Sn)
      .value("Sb", ElementType::Sb)
      .value("Te", ElementType::Te)
      .value("I", ElementType::I)
      .value("Xe", ElementType::Xe)
      .value("Cs", ElementType::Cs)
      .value("Ba", ElementType::Ba)
      .value("La", ElementType::La)
      .value("Ce", ElementType::Ce)
      .value("Pr", ElementType::Pr)
      .value("Nd", ElementType::Nd)
      .value("Pm", ElementType::Pm)
      .value("Sm", ElementType::Sm)
      .value("Eu", ElementType::Eu)
      .value("Gd", ElementType::Gd)
      .value("Tb", ElementType::Tb)
      .value("Dy", ElementType::Dy)
      .value("Ho", ElementType::Ho)
      .value("Er", ElementType::Er)
      .value("Tm", ElementType::Tm)
      .value("Yb", ElementType::Yb)
      .value("Lu", ElementType::Lu)
      .value("Hf", ElementType::Hf)
      .value("Ta", ElementType::Ta)
      .value("W", ElementType::W)
      .value("Re", ElementType::Re)
      .value("Os", ElementType::Os)
      .value("Ir", ElementType::Ir)
      .value("Pt", ElementType::Pt)
      .value("Au", ElementType::Au)
      .value("Hg", ElementType::Hg)
      .value("Tl", ElementType::Tl)
      .value("Pb", ElementType::Pb)
      .value("Bi", ElementType::Bi)
      .value("Po", ElementType::Po)
      .value("At", ElementType::At)
      .value("Rn", ElementType::Rn)
      .value("Fr", ElementType::Fr)
      .value("Ra", ElementType::Ra)
      .value("Ac", ElementType::Ac)
      .value("Th", ElementType::Th)
      .value("Pa", ElementType::Pa)
      .value("U", ElementType::U)
      .value("Np", ElementType::Np)
      .value("Pu", ElementType::Pu)
      .value("Am", ElementType::Am)
      .value("Cm", ElementType::Cm)
      .value("Bk", ElementType::Bk)
      .value("Cf", ElementType::Cf)
      .value("Es", ElementType::Es)
      .value("Fm", ElementType::Fm)
      .value("Md", ElementType::Md)
      .value("No", ElementType::No)
      .value("Lr", ElementType::Lr)
      .value("Rf", ElementType::Rf)
      .value("Db", ElementType::Db)
      .value("Sg", ElementType::Sg)
      .value("Bh", ElementType::Bh)
      .value("Hs", ElementType::Hs)
      .value("Mt", ElementType::Mt)
      .value("Ds", ElementType::Ds)
      .value("Rg", ElementType::Rg)
      .value("Cn", ElementType::Cn)
      .value("H1", ElementType::H1)
      .value("D", ElementType::D)
      .value("T", ElementType::T)
      .value("He3", ElementType::He3)
      .value("He4", ElementType::He4)
      .value("Li6", ElementType::Li6)
      .value("Li7", ElementType::Li7)
      .value("Be9", ElementType::Be9)
      .value("B10", ElementType::B10)
      .value("B11", ElementType::B11)
      .value("C12", ElementType::C12)
      .value("C13", ElementType::C13)
      .value("C14", ElementType::C14)
      .value("N14", ElementType::N14)
      .value("N15", ElementType::N15)
      .value("O16", ElementType::O16)
      .value("O17", ElementType::O17)
      .value("O18", ElementType::O18)
      .value("F19", ElementType::F19)
      .value("Ne20", ElementType::Ne20)
      .value("Ne21", ElementType::Ne21)
      .value("Ne22", ElementType::Ne22)
      .value("Na23", ElementType::Na23)
      .value("Mg24", ElementType::Mg24)
      .value("Mg25", ElementType::Mg25)
      .value("Mg26", ElementType::Mg26)
      .value("Al27", ElementType::Al27)
      .value("Si28", ElementType::Si28)
      .value("Si29", ElementType::Si29)
      .value("Si30", ElementType::Si30)
      .value("P31", ElementType::P31)
      .value("S32", ElementType::S32)
      .value("S33", ElementType::S33)
      .value("S34", ElementType::S34)
      .value("S36", ElementType::S36)
      .value("Cl35", ElementType::Cl35)
      .value("Cl37", ElementType::Cl37)
      .value("Ar36", ElementType::Ar36)
      .value("Ar38", ElementType::Ar38)
      .value("Ar40", ElementType::Ar40)
      .value("K39", ElementType::K39)
      .value("K40", ElementType::K40)
      .value("K41", ElementType::K41)
      .value("Ca40", ElementType::Ca40)
      .value("Ca42", ElementType::Ca42)
      .value("Ca43", ElementType::Ca43)
      .value("Ca44", ElementType::Ca44)
      .value("Ca46", ElementType::Ca46)
      .value("Ca48", ElementType::Ca48)
      .value("Sc45", ElementType::Sc45)
      .value("Ti46", ElementType::Ti46)
      .value("Ti47", ElementType::Ti47)
      .value("Ti48", ElementType::Ti48)
      .value("Ti49", ElementType::Ti49)
      .value("Ti50", ElementType::Ti50)
      .value("V50", ElementType::V50)
      .value("V51", ElementType::V51)
      .value("Cr50", ElementType::Cr50)
      .value("Cr52", ElementType::Cr52)
      .value("Cr53", ElementType::Cr53)
      .value("Cr54", ElementType::Cr54)
      .value("Mn55", ElementType::Mn55)
      .value("Fe54", ElementType::Fe54)
      .value("Fe56", ElementType::Fe56)
      .value("Fe57", ElementType::Fe57)
      .value("Fe58", ElementType::Fe58)
      .value("Co59", ElementType::Co59)
      .value("Ni58", ElementType::Ni58)
      .value("Ni60", ElementType::Ni60)
      .value("Ni61", ElementType::Ni61)
      .value("Ni62", ElementType::Ni62)
      .value("Ni64", ElementType::Ni64)
      .value("Cu63", ElementType::Cu63)
      .value("Cu65", ElementType::Cu65)
      .value("Zn64", ElementType::Zn64)
      .value("Zn66", ElementType::Zn66)
      .value("Zn67", ElementType::Zn67)
      .value("Zn68", ElementType::Zn68)
      .value("Zn70", ElementType::Zn70)
      .value("Ga69", ElementType::Ga69)
      .value("Ga71", ElementType::Ga71)
      .value("Ge70", ElementType::Ge70)
      .value("Ge72", ElementType::Ge72)
      .value("Ge73", ElementType::Ge73)
      .value("Ge74", ElementType::Ge74)
      .value("Ge76", ElementType::Ge76)
      .value("As75", ElementType::As75)
      .value("Se74", ElementType::Se74)
      .value("Se76", ElementType::Se76)
      .value("Se77", ElementType::Se77)
      .value("Se78", ElementType::Se78)
      .value("Se80", ElementType::Se80)
      .value("Se82", ElementType::Se82)
      .value("Br79", ElementType::Br79)
      .value("Br81", ElementType::Br81)
      .value("Kr78", ElementType::Kr78)
      .value("Kr80", ElementType::Kr80)
      .value("Kr82", ElementType::Kr82)
      .value("Kr83", ElementType::Kr83)
      .value("Kr84", ElementType::Kr84)
      .value("Kr86", ElementType::Kr86)
      .value("Rb85", ElementType::Rb85)
      .value("Rb87", ElementType::Rb87)
      .value("Sr84", ElementType::Sr84)
      .value("Sr86", ElementType::Sr86)
      .value("Sr87", ElementType::Sr87)
      .value("Sr88", ElementType::Sr88)
      .value("Y89", ElementType::Y89)
      .value("Zr90", ElementType::Zr90)
      .value("Zr91", ElementType::Zr91)
      .value("Zr92", ElementType::Zr92)
      .value("Zr94", ElementType::Zr94)
      .value("Zr96", ElementType::Zr96)
      .value("Nb93", ElementType::Nb93)
      .value("Mo92", ElementType::Mo92)
      .value("Mo94", ElementType::Mo94)
      .value("Mo95", ElementType::Mo95)
      .value("Mo96", ElementType::Mo96)
      .value("Mo97", ElementType::Mo97)
      .value("Mo98", ElementType::Mo98)
      .value("Mo100", ElementType::Mo100)
      .value("Tc97", ElementType::Tc97)
      .value("Tc98", ElementType::Tc98)
      .value("Tc99", ElementType::Tc99)
      .value("Ru96", ElementType::Ru96)
      .value("Ru98", ElementType::Ru98)
      .value("Ru99", ElementType::Ru99)
      .value("Ru100", ElementType::Ru100)
      .value("Ru101", ElementType::Ru101)
      .value("Ru102", ElementType::Ru102)
      .value("Ru104", ElementType::Ru104)
      .value("Rh103", ElementType::Rh103)
      .value("Pd102", ElementType::Pd102)
      .value("Pd104", ElementType::Pd104)
      .value("Pd105", ElementType::Pd105)
      .value("Pd106", ElementType::Pd106)
      .value("Pd108", ElementType::Pd108)
      .value("Pd110", ElementType::Pd110)
      .value("Ag107", ElementType::Ag107)
      .value("Ag109", ElementType::Ag109)
      .value("Cd106", ElementType::Cd106)
      .value("Cd108", ElementType::Cd108)
      .value("Cd110", ElementType::Cd110)
      .value("Cd111", ElementType::Cd111)
      .value("Cd112", ElementType::Cd112)
      .value("Cd113", ElementType::Cd113)
      .value("Cd114", ElementType::Cd114)
      .value("Cd116", ElementType::Cd116)
      .value("In113", ElementType::In113)
      .value("In115", ElementType::In115)
      .value("Sn112", ElementType::Sn112)
      .value("Sn114", ElementType::Sn114)
      .value("Sn115", ElementType::Sn115)
      .value("Sn116", ElementType::Sn116)
      .value("Sn117", ElementType::Sn117)
      .value("Sn118", ElementType::Sn118)
      .value("Sn119", ElementType::Sn119)
      .value("Sn120", ElementType::Sn120)
      .value("Sn122", ElementType::Sn122)
      .value("Sn124", ElementType::Sn124)
      .value("Sb121", ElementType::Sb121)
      .value("Sb123", ElementType::Sb123)
      .value("Te120", ElementType::Te120)
      .value("Te122", ElementType::Te122)
      .value("Te123", ElementType::Te123)
      .value("Te124", ElementType::Te124)
      .value("Te125", ElementType::Te125)
      .value("Te126", ElementType::Te126)
      .value("Te128", ElementType::Te128)
      .value("Te130", ElementType::Te130)
      .value("I127", ElementType::I127)
      .value("Xe124", ElementType::Xe124)
      .value("Xe126", ElementType::Xe126)
      .value("Xe128", ElementType::Xe128)
      .value("Xe129", ElementType::Xe129)
      .value("Xe130", ElementType::Xe130)
      .value("Xe131", ElementType::Xe131)
      .value("Xe132", ElementType::Xe132)
      .value("Xe134", ElementType::Xe134)
      .value("Xe136", ElementType::Xe136)
      .value("Cs133", ElementType::Cs133)
      .value("Ba130", ElementType::Ba130)
      .value("Ba132", ElementType::Ba132)
      .value("Ba134", ElementType::Ba134)
      .value("Ba135", ElementType::Ba135)
      .value("Ba136", ElementType::Ba136)
      .value("Ba137", ElementType::Ba137)
      .value("Ba138", ElementType::Ba138)
      .value("La138", ElementType::La138)
      .value("La139", ElementType::La139)
      .value("Ce136", ElementType::Ce136)
      .value("Ce138", ElementType::Ce138)
      .value("Ce140", ElementType::Ce140)
      .value("Ce142", ElementType::Ce142)
      .value("Pr141", ElementType::Pr141)
      .value("Nd142", ElementType::Nd142)
      .value("Nd143", ElementType::Nd143)
      .value("Nd144", ElementType::Nd144)
      .value("Nd145", ElementType::Nd145)
      .value("Nd146", ElementType::Nd146)
      .value("Nd148", ElementType::Nd148)
      .value("Nd150", ElementType::Nd150)
      .value("Pm145", ElementType::Pm145)
      .value("Pm147", ElementType::Pm147)
      .value("Sm144", ElementType::Sm144)
      .value("Sm147", ElementType::Sm147)
      .value("Sm148", ElementType::Sm148)
      .value("Sm149", ElementType::Sm149)
      .value("Sm150", ElementType::Sm150)
      .value("Sm152", ElementType::Sm152)
      .value("Sm154", ElementType::Sm154)
      .value("Eu151", ElementType::Eu151)
      .value("Eu153", ElementType::Eu153)
      .value("Gd152", ElementType::Gd152)
      .value("Gd154", ElementType::Gd154)
      .value("Gd155", ElementType::Gd155)
      .value("Gd156", ElementType::Gd156)
      .value("Gd157", ElementType::Gd157)
      .value("Gd158", ElementType::Gd158)
      .value("Gd160", ElementType::Gd160)
      .value("Tb159", ElementType::Tb159)
      .value("Dy156", ElementType::Dy156)
      .value("Dy158", ElementType::Dy158)
      .value("Dy160", ElementType::Dy160)
      .value("Dy161", ElementType::Dy161)
      .value("Dy162", ElementType::Dy162)
      .value("Dy163", ElementType::Dy163)
      .value("Dy164", ElementType::Dy164)
      .value("Ho165", ElementType::Ho165)
      .value("Er162", ElementType::Er162)
      .value("Er164", ElementType::Er164)
      .value("Er166", ElementType::Er166)
      .value("Er167", ElementType::Er167)
      .value("Er168", ElementType::Er168)
      .value("Er170", ElementType::Er170)
      .value("Tm169", ElementType::Tm169)
      .value("Yb168", ElementType::Yb168)
      .value("Yb170", ElementType::Yb170)
      .value("Yb171", ElementType::Yb171)
      .value("Yb172", ElementType::Yb172)
      .value("Yb173", ElementType::Yb173)
      .value("Yb174", ElementType::Yb174)
      .value("Yb176", ElementType::Yb176)
      .value("Lu175", ElementType::Lu175)
      .value("Lu176", ElementType::Lu176)
      .value("Hf174", ElementType::Hf174)
      .value("Hf176", ElementType::Hf176)
      .value("Hf177", ElementType::Hf177)
      .value("Hf178", ElementType::Hf178)
      .value("Hf179", ElementType::Hf179)
      .value("Hf180", ElementType::Hf180)
      .value("Ta180", ElementType::Ta180)
      .value("Ta181", ElementType::Ta181)
      .value("W180", ElementType::W180)
      .value("W182", ElementType::W182)
      .value("W183", ElementType::W183)
      .value("W184", ElementType::W184)
      .value("W186", ElementType::W186)
      .value("Re185", ElementType::Re185)
      .value("Re187", ElementType::Re187)
      .value("Os184", ElementType::Os184)
      .value("Os186", ElementType::Os186)
      .value("Os187", ElementType::Os187)
      .value("Os188", ElementType::Os188)
      .value("Os189", ElementType::Os189)
      .value("Os190", ElementType::Os190)
      .value("Os192", ElementType::Os192)
      .value("Ir191", ElementType::Ir191)
      .value("Ir193", ElementType::Ir193)
      .value("Pt190", ElementType::Pt190)
      .value("Pt192", ElementType::Pt192)
      .value("Pt194", ElementType::Pt194)
      .value("Pt195", ElementType::Pt195)
      .value("Pt196", ElementType::Pt196)
      .value("Pt198", ElementType::Pt198)
      .value("Au197", ElementType::Au197)
      .value("Hg196", ElementType::Hg196)
      .value("Hg198", ElementType::Hg198)
      .value("Hg199", ElementType::Hg199)
      .value("Hg200", ElementType::Hg200)
      .value("Hg201", ElementType::Hg201)
      .value("Hg202", ElementType::Hg202)
      .value("Hg204", ElementType::Hg204)
      .value("Tl203", ElementType::Tl203)
      .value("Tl205", ElementType::Tl205)
      .value("Pb204", ElementType::Pb204)
      .value("Pb206", ElementType::Pb206)
      .value("Pb207", ElementType::Pb207)
      .value("Pb208", ElementType::Pb208)
      .value("Bi209", ElementType::Bi209)
      .value("Po209", ElementType::Po209)
      .value("Po210", ElementType::Po210)
      .value("At210", ElementType::At210)
      .value("At211", ElementType::At211)
      .value("Rn211", ElementType::Rn211)
      .value("Rn220", ElementType::Rn220)
      .value("Rn222", ElementType::Rn222)
      .value("Fr223", ElementType::Fr223)
      .value("Ra223", ElementType::Ra223)
      .value("Ra224", ElementType::Ra224)
      .value("Ra226", ElementType::Ra226)
      .value("Ra228", ElementType::Ra228)
      .value("Ac227", ElementType::Ac227)
      .value("Th230", ElementType::Th230)
      .value("Th232", ElementType::Th232)
      .value("Pa231", ElementType::Pa231)
      .value("U233", ElementType::U233)
      .value("U234", ElementType::U234)
      .value("U235", ElementType::U235)
      .value("U236", ElementType::U236)
      .value("U238", ElementType::U238)
      .value("Np236", ElementType::Np236)
      .value("Np237", ElementType::Np237)
      .value("Pu238", ElementType::Pu238)
      .value("Pu239", ElementType::Pu239)
      .value("Pu240", ElementType::Pu240)
      .value("Pu241", ElementType::Pu241)
      .value("Pu242", ElementType::Pu242)
      .value("Pu244", ElementType::Pu244)
      .value("Am241", ElementType::Am241)
      .value("Am243", ElementType::Am243)
      .value("Cm243", ElementType::Cm243)
      .value("Cm244", ElementType::Cm244)
      .value("Cm245", ElementType::Cm245)
      .value("Cm246", ElementType::Cm246)
      .value("Cm247", ElementType::Cm247)
      .value("Cm248", ElementType::Cm248)
      .value("Bk247", ElementType::Bk247)
      .value("Bk249", ElementType::Bk249)
      .value("Cf249", ElementType::Cf249)
      .value("Cf250", ElementType::Cf250)
      .value("Cf251", ElementType::Cf251)
      .value("Cf252", ElementType::Cf252)
      .value("Es252", ElementType::Es252)
      .value("Fm257", ElementType::Fm257)
      .value("Md258", ElementType::Md258)
      .value("Md260", ElementType::Md260)
      .value("No259", ElementType::No259)
      .value("Lr262", ElementType::Lr262)
      .value("Rf267", ElementType::Rf267)
      .value("Db268", ElementType::Db268)
      .value("Sg271", ElementType::Sg271)
      .value("Bh272", ElementType::Bh272)
      .value("Hs270", ElementType::Hs270)
      .value("Mt276", ElementType::Mt276)
      .value("Ds281", ElementType::Ds281)
      .value("Rg280", ElementType::Rg280)
      .value("Cn285", ElementType::Cn285);

  element_type.def(
      "__repr__",
      [](const pybind11::object o) -> std::string {
        // Generate a qualified literal for the enum: ElementType.H1
        auto typehandle = pybind11::type::handle_of(o);
        auto qualifiedName =
            typehandle.attr("__qualname__").cast<std::string>() + "." + o.attr("name")().cast<std::string>();
        return qualifiedName;
      },
      pybind11::prepend());

  element_type.def(
      "__str__",
      [](const pybind11::object o) -> std::string {
        /* Since we have no function to string-represent isotopes, we're going to
         * just lookup the enum name in its members dictionary. Not particularly
         * efficient.
         */
        ElementType e = o.cast<ElementType>();
        // Get the members dict: str -> Utils.ElementType
        pybind11::dict dict = o.attr("__members__");
        for (auto pair : dict) {
          if (pair.second.cast<ElementType>() == e) {
            return pair.first.cast<std::string>();
          }
        }
        throw std::runtime_error("No such element type");
      },
      pybind11::prepend());
}
