rule [
   ruleID "2"
   left [
      edge [ source 1 target 2 label "-" ]
      edge [ source 3 target 4 label "-" ]
   ]
   context [
      node [ id 1 label "C" ]
      node [ id 2 label "Cl" ]
      node [ id 3 label "N" ]
      node [ id 4 label "H" ]
   ]
   right [
      edge [ source 1 target 3 label "-" ]
      edge [ source 2 target 4 label "-" ]
   ]
]