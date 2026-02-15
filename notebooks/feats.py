feats = [
    "Aeration rate(Fg:L/h)", # step-wise signal
    "Sugar feed rate(Fs:L/h)",  # step-wise signal
    "Acid flow rate(Fa:L/h)", # spiky sparse signal
    "Base flow rate(Fb:L/h)", # noisy with trend signal
    "Heating/cooling water flow rate(Fc:L/h)", # noisy with trend signal
    "Heating water flow rate(Fh:L/h)", # spiky sparse signal
    "Water for injection/dilution(Fw:L/h)", # step-wise signal
    "Air head pressure(pressure:bar)", # step-wise signal
    "Dumped broth flow(Fremoved:L/h)", # spiky sparse signal
    "Substrate concentration(S:g/L)", # noisy with trend signal
    "Dissolved oxygen concentration(DO2:mg/L)", # trend signal with sudden jumps
    "Penicillin concentration(P:g/L)", # smooth signal
    "Vessel Volume(V:L)", # trend signal with sudden jumps
    "Vessel Weight(Wt:Kg)", # trend signal with sudden jumps
    "pH(pH:pH)", # noisy with short trend signal
    "Temperature(T:K)", # noisy with short trend signal
    "Generated heat(Q:kJ)", # noisy with short trend signal
    "carbon dioxide percent in off-gas(CO2outgas:%)", # trend signal with sudden jumps
    "PAA flow(Fpaa:PAA flow (L/h))", # step-wise signal
    "PAA concentration offline(PAA_offline:PAA (g L^{-1}))", # smooth signal not available at all timepoints (nan where not)
    "Oil flow(Foil:L/hr)", # step-wise signal
    "NH_3 concentration off-line(NH3_offline:NH3 (g L^{-1}))", # smooth signal not available at all timepoints (nan where not)
    "Oxygen Uptake Rate(OUR:(g min^{-1}))", # noisy with trend signal
    "Oxygen in percent in off-gas(O2:O2  (%))", # noisy with trend signal
    "Offline Penicillin concentration(P_offline:P(g L^{-1}))", # smooth signal not available at all timepoints (nan where not)
    "Offline Biomass concentratio(X_offline:X(g L^{-1}))", # smooth signal not available at all timepoints (nan where not)
    "Carbon evolution rate(CER:g/h)", # trend signal with sudden jumps
    "Viscosity(Viscosity_offline:centPoise)" # smooth signal not available at all timepoints (nan where not)
    ]