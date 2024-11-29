debug = True

predicate_names = ["cobot_holds", "human_holds", "human_glues", "qc"]
predicate_names = ["a", "b" , "c"]

# Each trace is a list of events, each event is a list of predicate values. Use the value 0 for predicates that were not logged in an event.
traces = []
traces.append( [[0.11, 0.12, 0.13, 0.14], [0.21, 0.22, 0.23, 0.24], [0.31, 0.32,  0.33, 0.34,], [0.41, 0.42, 0.43, 0.44]] ) 
traces.append( [[0.08, 0.3, 0.1, 0.9], [0.9, 0.9, 0.93, 0.94], [0.91, 0.03, 0.4, 0.72]] )
traces.append( [[0.14, 0.3, 0.4, 0.23], [0.2, 0.07, 0.4, 0.14], [0.2, 0.93, 0.7, 0.82], [0.31, 0.03, 0.42, 0.72], [0.33, 0.03, 0.41, 0.92], [0.12, 0.63, 0.03, 0.41]] )
traces.append( [[0.2, 1, 0.3, 1], [0.6, 1, 0.43, 1], [0.6, 1, 0.4, 1]] )
traces.append( [[0.3, 0, 0, 0], [0.51, 0.32, 0.89, 0.72]] )


#the formula
formula = "G(a -> X(b))"
formula = "G((cobot_holds && human_holds) <-> X(human_glues))"
formula = "(human_glues U cobot_holds)" 
formula = "S(cobot_holds)" 
formula = "BF[61](cobot_holds & human_glues)"
formula = "(human_glues BU[11] cobot_holds)" 
formula = "S(cobot_holds)"
formula = "FG(cobot_holds)" 
formula = "F XX(cobot_holds)" 
formula = "IN[34](cobot_holds & human_glues)"
formula = "AG(X(cobot_holds))" 
formula = "X AG(cobot_holds)"
formula = "X BAG[1](cobot_holds)"
formula = "human_glues AU cobot_holds"
formula = "L[3](cobot_holds)"
formula = "AG(cobot_holds)" 
formula = "(human_glues AU cobot_holds) | AG(cobot_holds)" 
formula = "BAG[3](cobot_holds)" 
formula = "L[2](cobot_holds)" 

#with comparisons
#use parentheses around subformulas
#a float allowed on the right-hand side only
formula = "(F(cobot_holds)) > human_glues" 
formula = "(F(cobot_holds)) > (G(human_glues))" 
formula = "(F(cobot_holds)) > 0.5" 
formula = "(F(cobot_holds > 0.5)) == 1" 
formula = "(G(cobot_holds > 0.1)) & F cobot_holds"
formula = "(G(cobot_holds > 0.2))" 
formula = "F(((G(cobot_holds > 0.1)) & F cobot_holds) > 0.6)"

#the instant
i=0


formula = "F(((!(cobot_holds)U(human_glues)) | G((((!(qc)U(cobot_holds)) | G(!(human_holds)))) != (G(F(qc) -> !(cobot_holds))))) == (F(human_glues) -> F(qc)) )" 
