clear *

cls

import delimited "C:\The Shop\china_import_final.csv", clear 

rename (treated v1) (c0 time)

g time2 = _n

qui reshape long c, i(time time2) j(id)
replace id =id+1
xtset id time2

tempvar unit

g `unit' = "Unit "

egen unit = concat(`unit' id)

g treat = cond(id==1 & time2 >=36,1,0)

drop time

rename (time c) (time import)

qui fdid import, tr(treat) unitnames(unit)

// Controls FDID Selects: Unit 61, Unit 83, Unit 27, Unit 59, Unit 46, Unit 51,
// Unit 16, Unit 20, Unit 22, Unit 55, Unit 82, Unit 32, Unit 86, Unit 52
// (14 in total)

// Zhentao's data do not allow me to see the donor names. 
// fsPDA selects C60, 45, 25
// FDID selects all of these.


// R code:

// ATT = -0.0309

// SE: 0.0274

// T = -1.1262

// P = 0.2601

// R2 = 0.777

mkf newframe
cwf newframe
svmat e(series), names(col)
replace cf1 = cf1*100
replace import1 = import1*100


loc cfcol2 blue
loc cfcol black

twoway (connected import1 time, ///
	mcolor(red) msize(small) msymbol(smcircle) lcolor(red) lwidth(medium)) ///
(connected cf1 time if time < 36, ///
	mcolor(`cfcol') msize(medsmall) msymbol(smtriangle) ///
	lcolor(`cfcol') lpattern(shortdash) lwidth(medium)) ///
(connected cf1 time if time >=36, ///
	mcolor(`cfcol2') msymbol(smsquare) ///
	lcolor(`cfcol2') lpattern(shortdash) lwidth(medium)), ///
	note("Jan of 2013 is time point 36.") ///
	legend(order(1 "Observation" 2 "In-Sample Fit" 3 "Counterfactual") ///
	ring(0) pos(7) region(fcolor(none))) yti("Import Growth Rate (%)") ///
	ylabel(#5, grid glwidth(vthin) glcolor(gs4%20) glpattern(solid)) ///
	xline(36, lwidth(medium) lpattern(solid) lcolor(black)) ///
	xlabel(#10, grid glwidth(vthin) glcolor(gs4%20) glpattern(solid)) ///
	xti("Time")   //plotregion(fcolor(gs6) lcolor(gs6) ifcolor(gs6) ilcolor(gs6))
