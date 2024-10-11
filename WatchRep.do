clear *

// ssc install rcm
//net install sdid_event, from("https://raw.githubusercontent.com/DiegoCiccia/sdid/main/sdid_event") replace
// needed for FDID ^, even though it is not used here
//net inst fdid, from("https://raw.githubusercontent.com/jgreathouse9/FDIDTutorial/main") replace

cls // clears the screen

import delimited "C:\The Shop\china_import_final.csv", clear
// imports data that was created in R

rename (treated v1) (c0 time)


g time2 = _n
// making a numeric time variable (the data are wide at present)

qui reshape long c, i(time time2) j(id)
// makes it long

replace id =id+1
// id = +1, so all control units msut be interpreted as
// id -1

xtset id time2
// declares panel data

tempvar unit

g `unit' = "Unit "

egen unit = concat(`unit' id)
// FDID demands that the units be labeled somehow
// so I just assign them "Unit X"

g treat = cond(id==1 & time2 >=36,1,0)

// where 36 = Jan of 2013

drop time

rename (time c) (time import)
// renames our variables to relevant names

//replace import = import * 100

cls // clears our screen again

// Regression Control Method
rcm import, trperiod(36) trunit(1) method(forward)  frame(fsframe) criterion(mbic)

// rcm import, trperiod(36) trunit(1) method(forward)  frame(fsframe)

// Here I grab the counterfactual
// of RCM to use in the plot


// entering fsframe ...
frame fsframe {

keep time pred·import·1

rename pred·import·1 cf_fspda

} // leaving fsframe ...

// estimates FDID, using the default standard error as per
// Kathy's original MATLAB code

qui fdid import, tr(treat) unitnames(unit)
/*
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
*/

// We make a new frame to place these results

mkf newframe
cwf newframe

// Using e(series), a stored matrix of fdid

svmat e(series), names(col)


// our treatment date
loc interdate: di tm(2013m1)


// creating the real calendar time with event time
g monthtime= `interdate' + eventtime1


// now we tsset this dataset...

tsset monthtime, m

// merge in the RCM counterfactual from the other frame...
qui frlink 1:1 time, frame(fsframe)

qui frget cf_fspda, from(fsframe)
drop fsframe

cls
loc interdate: di tm(2013m1)

// And here is the plot!
twoway (tsline import1 , recast(connected) ///
	mcolor(red) msymbol(smcircle) lcolor(red) lwidth(medium)) ///
(connected cf1 monthtime, ///
	mcolor(black) msize(medsmall) msymbol(smtriangle) ///
	lcolor(black) lpattern(shortdash) lwidth(thin)) ///
(connected cf_fspda monthtime, ///
	mcolor("0 173 255") msymbol(smsquare) ///
	lcolor("0 173 255") lpattern(shortdash) lwidth(thin)), ///
	legend(order(1 "Observation" 2 "fDID" 3 "RCM") ///
	pos(6) region(fcolor(none)) rows(1)) yti("Import Growth Rate (%)") ///
	ylabel(-.6(.2).6, grid glwidth(vthin) glcolor(gs11%10) glpattern(solid)) ///
	xline(`interdate', lwidth(medium) lpattern(solid) lcolor(black)) ///
	xti("Year") xsize(9.5) ysize(6) ///
	plotregion(margin(zero)) tmtick(##1) ///
	tlabel(#4,  angle(forty_five) format(%-tmNN/CCYY) grid glwidth(vthin) glcolor(gs11%10) glpattern(solid))
	
graph export "C:\The Shop\FDIDPlotWatch.eps", as(eps) name("Graph") preview(on) replace
