#include "lib_quadrature.h"
#include "gauss_internal.h"

/*
	Numerical Integration by Gauss-Legendre Quadrature Formulas of high
   orders.
	High-precision abscissas and weights are used.

	Project homepage: http://www.holoborodko.com/pavel/?page_id=679
	Contact e-mail:   pavel@holoborodko.com

	Copyright (c)2007-2010 Pavel Holoborodko
	All rights reserved.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions
	are met:

	1. Redistributions of source code must retain the above copyright
	notice, this list of conditions and the following disclaimer.

	2. Redistributions in binary form must reproduce the above copyright
	notice, this list of conditions and the following disclaimer in the
	documentation and/or other materials provided with the distribution.

	3. Redistributions of any form whatsoever must retain the following
	acknowledgment:
	"
	 This product includes software developed by Pavel Holoborodko
	 Web: http://www.holoborodko.com/pavel/
	 e-mail: pavel@holoborodko.com

	"

	4. This software cannot be, by any means, used for any commercial
	purpose without the prior permission of the copyright holder.

	Any of the above conditions can be waived if you get permission from
	the copyright holder.

	THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
	IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
   PURPOSE
	ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
	FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL
	DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
	OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
	HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
   STRICT
	LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
   WAY
	OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
	SUCH DAMAGE.

	Contributors
	Konstantin Holoborodko - Optimization of Legendre polynomial computing.

*/

#ifndef PI
#define PI 3.1415926535897932384626433832795028841971693993751
#endif

#ifndef FABS
#define FABS(a) ((a) >= 0 ? (a) : -(a))
#endif

static double ltbl[1024]
    = {0.00000000000000000000, 0.00000000000000000000, 0.50000000000000000000, 0.66666666666666674000,
       0.75000000000000000000, 0.80000000000000004000, 0.83333333333333337000, 0.85714285714285721000,
       0.87500000000000000000, 0.88888888888888884000, 0.90000000000000002000, 0.90909090909090906000,
       0.91666666666666663000, 0.92307692307692313000, 0.92857142857142860000, 0.93333333333333335000,
       0.93750000000000000000, 0.94117647058823528000, 0.94444444444444442000, 0.94736842105263164000,
       0.94999999999999996000, 0.95238095238095233000, 0.95454545454545459000, 0.95652173913043481000,
       0.95833333333333337000, 0.95999999999999996000, 0.96153846153846156000, 0.96296296296296302000,
       0.96428571428571430000, 0.96551724137931039000, 0.96666666666666667000, 0.96774193548387100000,
       0.96875000000000000000, 0.96969696969696972000, 0.97058823529411764000, 0.97142857142857142000,
       0.97222222222222221000, 0.97297297297297303000, 0.97368421052631582000, 0.97435897435897434000,
       0.97499999999999998000, 0.97560975609756095000, 0.97619047619047616000, 0.97674418604651159000,
       0.97727272727272729000, 0.97777777777777775000, 0.97826086956521741000, 0.97872340425531912000,
       0.97916666666666663000, 0.97959183673469385000, 0.97999999999999998000, 0.98039215686274506000,
       0.98076923076923073000, 0.98113207547169812000, 0.98148148148148151000, 0.98181818181818181000,
       0.98214285714285710000, 0.98245614035087714000, 0.98275862068965514000, 0.98305084745762716000,
       0.98333333333333328000, 0.98360655737704916000, 0.98387096774193550000, 0.98412698412698418000,
       0.98437500000000000000, 0.98461538461538467000, 0.98484848484848486000, 0.98507462686567160000,
       0.98529411764705888000, 0.98550724637681164000, 0.98571428571428577000, 0.98591549295774650000,
       0.98611111111111116000, 0.98630136986301364000, 0.98648648648648651000, 0.98666666666666669000,
       0.98684210526315785000, 0.98701298701298701000, 0.98717948717948723000, 0.98734177215189878000,
       0.98750000000000004000, 0.98765432098765427000, 0.98780487804878048000, 0.98795180722891562000,
       0.98809523809523814000, 0.98823529411764710000, 0.98837209302325579000, 0.98850574712643680000,
       0.98863636363636365000, 0.98876404494382020000, 0.98888888888888893000, 0.98901098901098905000,
       0.98913043478260865000, 0.98924731182795700000, 0.98936170212765961000, 0.98947368421052628000,
       0.98958333333333337000, 0.98969072164948457000, 0.98979591836734693000, 0.98989898989898994000,
       0.98999999999999999000, 0.99009900990099009000, 0.99019607843137258000, 0.99029126213592233000,
       0.99038461538461542000, 0.99047619047619051000, 0.99056603773584906000, 0.99065420560747663000,
       0.99074074074074070000, 0.99082568807339455000, 0.99090909090909096000, 0.99099099099099097000,
       0.99107142857142860000, 0.99115044247787609000, 0.99122807017543857000, 0.99130434782608701000,
       0.99137931034482762000, 0.99145299145299148000, 0.99152542372881358000, 0.99159663865546221000,
       0.99166666666666670000, 0.99173553719008267000, 0.99180327868852458000, 0.99186991869918695000,
       0.99193548387096775000, 0.99199999999999999000, 0.99206349206349209000, 0.99212598425196852000,
       0.99218750000000000000, 0.99224806201550386000, 0.99230769230769234000, 0.99236641221374045000,
       0.99242424242424243000, 0.99248120300751874000, 0.99253731343283580000, 0.99259259259259258000,
       0.99264705882352944000, 0.99270072992700731000, 0.99275362318840576000, 0.99280575539568350000,
       0.99285714285714288000, 0.99290780141843971000, 0.99295774647887325000, 0.99300699300699302000,
       0.99305555555555558000, 0.99310344827586206000, 0.99315068493150682000, 0.99319727891156462000,
       0.99324324324324320000, 0.99328859060402686000, 0.99333333333333329000, 0.99337748344370858000,
       0.99342105263157898000, 0.99346405228758172000, 0.99350649350649356000, 0.99354838709677418000,
       0.99358974358974361000, 0.99363057324840764000, 0.99367088607594933000, 0.99371069182389937000,
       0.99375000000000002000, 0.99378881987577639000, 0.99382716049382713000, 0.99386503067484666000,
       0.99390243902439024000, 0.99393939393939390000, 0.99397590361445787000, 0.99401197604790414000,
       0.99404761904761907000, 0.99408284023668636000, 0.99411764705882355000, 0.99415204678362579000,
       0.99418604651162790000, 0.99421965317919070000, 0.99425287356321834000, 0.99428571428571433000,
       0.99431818181818177000, 0.99435028248587576000, 0.99438202247191010000, 0.99441340782122900000,
       0.99444444444444446000, 0.99447513812154698000, 0.99450549450549453000, 0.99453551912568305000,
       0.99456521739130432000, 0.99459459459459465000, 0.99462365591397850000, 0.99465240641711228000,
       0.99468085106382975000, 0.99470899470899465000, 0.99473684210526314000, 0.99476439790575921000,
       0.99479166666666663000, 0.99481865284974091000, 0.99484536082474229000, 0.99487179487179489000,
       0.99489795918367352000, 0.99492385786802029000, 0.99494949494949492000, 0.99497487437185927000,
       0.99500000000000000000, 0.99502487562189057000, 0.99504950495049505000, 0.99507389162561577000,
       0.99509803921568629000, 0.99512195121951219000, 0.99514563106796117000, 0.99516908212560384000,
       0.99519230769230771000, 0.99521531100478466000, 0.99523809523809526000, 0.99526066350710896000,
       0.99528301886792447000, 0.99530516431924887000, 0.99532710280373837000, 0.99534883720930234000,
       0.99537037037037035000, 0.99539170506912444000, 0.99541284403669728000, 0.99543378995433796000,
       0.99545454545454548000, 0.99547511312217196000, 0.99549549549549554000, 0.99551569506726456000,
       0.99553571428571430000, 0.99555555555555553000, 0.99557522123893805000, 0.99559471365638763000,
       0.99561403508771928000, 0.99563318777292575000, 0.99565217391304350000, 0.99567099567099571000,
       0.99568965517241381000, 0.99570815450643779000, 0.99572649572649574000, 0.99574468085106382000,
       0.99576271186440679000, 0.99578059071729963000, 0.99579831932773111000, 0.99581589958159000000,
       0.99583333333333335000, 0.99585062240663902000, 0.99586776859504134000, 0.99588477366255146000,
       0.99590163934426235000, 0.99591836734693873000, 0.99593495934959353000, 0.99595141700404854000,
       0.99596774193548387000, 0.99598393574297184000, 0.99600000000000000000, 0.99601593625498008000,
       0.99603174603174605000, 0.99604743083003955000, 0.99606299212598426000, 0.99607843137254903000,
       0.99609375000000000000, 0.99610894941634243000, 0.99612403100775193000, 0.99613899613899615000,
       0.99615384615384617000, 0.99616858237547889000, 0.99618320610687028000, 0.99619771863117867000,
       0.99621212121212122000, 0.99622641509433962000, 0.99624060150375937000, 0.99625468164794007000,
       0.99626865671641796000, 0.99628252788104088000, 0.99629629629629635000, 0.99630996309963105000,
       0.99632352941176472000, 0.99633699633699635000, 0.99635036496350360000, 0.99636363636363634000,
       0.99637681159420288000, 0.99638989169675085000, 0.99640287769784175000, 0.99641577060931896000,
       0.99642857142857144000, 0.99644128113879005000, 0.99645390070921991000, 0.99646643109540634000,
       0.99647887323943662000, 0.99649122807017543000, 0.99650349650349646000, 0.99651567944250874000,
       0.99652777777777779000, 0.99653979238754320000, 0.99655172413793103000, 0.99656357388316152000,
       0.99657534246575341000, 0.99658703071672350000, 0.99659863945578231000, 0.99661016949152548000,
       0.99662162162162160000, 0.99663299663299665000, 0.99664429530201337000, 0.99665551839464883000,
       0.99666666666666670000, 0.99667774086378735000, 0.99668874172185429000, 0.99669966996699666000,
       0.99671052631578949000, 0.99672131147540988000, 0.99673202614379086000, 0.99674267100977199000,
       0.99675324675324672000, 0.99676375404530748000, 0.99677419354838714000, 0.99678456591639875000,
       0.99679487179487181000, 0.99680511182108622000, 0.99681528662420382000, 0.99682539682539684000,
       0.99683544303797467000, 0.99684542586750791000, 0.99685534591194969000, 0.99686520376175547000,
       0.99687499999999996000, 0.99688473520249221000, 0.99689440993788825000, 0.99690402476780182000,
       0.99691358024691357000, 0.99692307692307691000, 0.99693251533742333000, 0.99694189602446481000,
       0.99695121951219512000, 0.99696048632218848000, 0.99696969696969695000, 0.99697885196374625000,
       0.99698795180722888000, 0.99699699699699695000, 0.99700598802395213000, 0.99701492537313430000,
       0.99702380952380953000, 0.99703264094955490000, 0.99704142011834318000, 0.99705014749262533000,
       0.99705882352941178000, 0.99706744868035191000, 0.99707602339181289000, 0.99708454810495628000,
       0.99709302325581395000, 0.99710144927536237000, 0.99710982658959535000, 0.99711815561959649000,
       0.99712643678160917000, 0.99713467048710602000, 0.99714285714285711000, 0.99715099715099720000,
       0.99715909090909094000, 0.99716713881019825000, 0.99717514124293782000, 0.99718309859154930000,
       0.99719101123595510000, 0.99719887955182074000, 0.99720670391061450000, 0.99721448467966578000,
       0.99722222222222223000, 0.99722991689750695000, 0.99723756906077343000, 0.99724517906336085000,
       0.99725274725274726000, 0.99726027397260275000, 0.99726775956284153000, 0.99727520435967298000,
       0.99728260869565222000, 0.99728997289972898000, 0.99729729729729732000, 0.99730458221024254000,
       0.99731182795698925000, 0.99731903485254692000, 0.99732620320855614000, 0.99733333333333329000,
       0.99734042553191493000, 0.99734748010610075000, 0.99735449735449733000, 0.99736147757255933000,
       0.99736842105263157000, 0.99737532808398954000, 0.99738219895287961000, 0.99738903394255873000,
       0.99739583333333337000, 0.99740259740259740000, 0.99740932642487046000, 0.99741602067183466000,
       0.99742268041237114000, 0.99742930591259638000, 0.99743589743589745000, 0.99744245524296671000,
       0.99744897959183676000, 0.99745547073791352000, 0.99746192893401020000, 0.99746835443037973000,
       0.99747474747474751000, 0.99748110831234260000, 0.99748743718592969000, 0.99749373433583965000,
       0.99750000000000005000, 0.99750623441396513000, 0.99751243781094523000, 0.99751861042183620000,
       0.99752475247524752000, 0.99753086419753090000, 0.99753694581280783000, 0.99754299754299758000,
       0.99754901960784315000, 0.99755501222493892000, 0.99756097560975610000, 0.99756690997566910000,
       0.99757281553398058000, 0.99757869249394671000, 0.99758454106280192000, 0.99759036144578317000,
       0.99759615384615385000, 0.99760191846522783000, 0.99760765550239239000, 0.99761336515513122000,
       0.99761904761904763000, 0.99762470308788598000, 0.99763033175355453000, 0.99763593380614657000,
       0.99764150943396224000, 0.99764705882352944000, 0.99765258215962438000, 0.99765807962529274000,
       0.99766355140186913000, 0.99766899766899764000, 0.99767441860465111000, 0.99767981438515085000,
       0.99768518518518523000, 0.99769053117782913000, 0.99769585253456217000, 0.99770114942528731000,
       0.99770642201834858000, 0.99771167048054921000, 0.99771689497716898000, 0.99772209567198178000,
       0.99772727272727268000, 0.99773242630385484000, 0.99773755656108598000, 0.99774266365688491000,
       0.99774774774774777000, 0.99775280898876406000, 0.99775784753363228000, 0.99776286353467558000,
       0.99776785714285710000, 0.99777282850779514000, 0.99777777777777776000, 0.99778270509977829000,
       0.99778761061946908000, 0.99779249448123619000, 0.99779735682819382000, 0.99780219780219781000,
       0.99780701754385970000, 0.99781181619256021000, 0.99781659388646293000, 0.99782135076252720000,
       0.99782608695652175000, 0.99783080260303691000, 0.99783549783549785000, 0.99784017278617709000,
       0.99784482758620685000, 0.99784946236559136000, 0.99785407725321884000, 0.99785867237687365000,
       0.99786324786324787000, 0.99786780383795304000, 0.99787234042553197000, 0.99787685774946921000,
       0.99788135593220340000, 0.99788583509513740000, 0.99789029535864981000, 0.99789473684210528000,
       0.99789915966386555000, 0.99790356394129975000, 0.99790794979079500000, 0.99791231732776620000,
       0.99791666666666667000, 0.99792099792099798000, 0.99792531120331951000, 0.99792960662525876000,
       0.99793388429752061000, 0.99793814432989691000, 0.99794238683127567000, 0.99794661190965095000,
       0.99795081967213117000, 0.99795501022494892000, 0.99795918367346936000, 0.99796334012219956000,
       0.99796747967479671000, 0.99797160243407712000, 0.99797570850202433000, 0.99797979797979797000,
       0.99798387096774188000, 0.99798792756539234000, 0.99799196787148592000, 0.99799599198396793000,
       0.99800000000000000000, 0.99800399201596801000, 0.99800796812749004000, 0.99801192842942343000,
       0.99801587301587302000, 0.99801980198019802000, 0.99802371541501977000, 0.99802761341222879000,
       0.99803149606299213000, 0.99803536345776034000, 0.99803921568627452000, 0.99804305283757344000,
       0.99804687500000000000, 0.99805068226120852000, 0.99805447470817121000, 0.99805825242718449000,
       0.99806201550387597000, 0.99806576402321079000, 0.99806949806949807000, 0.99807321772639690000,
       0.99807692307692308000, 0.99808061420345484000, 0.99808429118773945000, 0.99808795411089868000,
       0.99809160305343514000, 0.99809523809523815000, 0.99809885931558939000, 0.99810246679316883000,
       0.99810606060606055000, 0.99810964083175802000, 0.99811320754716981000, 0.99811676082862522000,
       0.99812030075187974000, 0.99812382739212002000, 0.99812734082397003000, 0.99813084112149530000,
       0.99813432835820892000, 0.99813780260707630000, 0.99814126394052050000, 0.99814471243042668000,
       0.99814814814814812000, 0.99815157116451014000, 0.99815498154981552000, 0.99815837937384899000,
       0.99816176470588236000, 0.99816513761467895000, 0.99816849816849818000, 0.99817184643510060000,
       0.99817518248175185000, 0.99817850637522765000, 0.99818181818181817000, 0.99818511796733211000,
       0.99818840579710144000, 0.99819168173598549000, 0.99819494584837543000, 0.99819819819819822000,
       0.99820143884892087000, 0.99820466786355477000, 0.99820788530465954000, 0.99821109123434704000,
       0.99821428571428572000, 0.99821746880570406000, 0.99822064056939497000, 0.99822380106571940000,
       0.99822695035460995000, 0.99823008849557526000, 0.99823321554770317000, 0.99823633156966496000,
       0.99823943661971826000, 0.99824253075571179000, 0.99824561403508771000, 0.99824868651488619000,
       0.99825174825174823000, 0.99825479930191974000, 0.99825783972125437000, 0.99826086956521742000,
       0.99826388888888884000, 0.99826689774696709000, 0.99826989619377160000, 0.99827288428324701000,
       0.99827586206896557000, 0.99827882960413084000, 0.99828178694158076000, 0.99828473413379071000,
       0.99828767123287676000, 0.99829059829059830000, 0.99829351535836175000, 0.99829642248722317000,
       0.99829931972789121000, 0.99830220713073003000, 0.99830508474576274000, 0.99830795262267347000,
       0.99831081081081086000, 0.99831365935919059000, 0.99831649831649827000, 0.99831932773109244000,
       0.99832214765100669000, 0.99832495812395305000, 0.99832775919732442000, 0.99833055091819700000,
       0.99833333333333329000, 0.99833610648918469000, 0.99833887043189373000, 0.99834162520729686000,
       0.99834437086092720000, 0.99834710743801658000, 0.99834983498349839000, 0.99835255354200991000,
       0.99835526315789469000, 0.99835796387520526000, 0.99836065573770494000, 0.99836333878887074000,
       0.99836601307189543000, 0.99836867862969003000, 0.99837133550488599000, 0.99837398373983743000,
       0.99837662337662336000, 0.99837925445705022000, 0.99838187702265369000, 0.99838449111470118000,
       0.99838709677419357000, 0.99838969404186795000, 0.99839228295819937000, 0.99839486356340290000,
       0.99839743589743590000, 0.99839999999999995000, 0.99840255591054317000, 0.99840510366826152000,
       0.99840764331210186000, 0.99841017488076311000, 0.99841269841269842000, 0.99841521394611732000,
       0.99841772151898733000, 0.99842022116903628000, 0.99842271293375395000, 0.99842519685039366000,
       0.99842767295597479000, 0.99843014128728413000, 0.99843260188087779000, 0.99843505477308292000,
       0.99843749999999998000, 0.99843993759750393000, 0.99844236760124616000, 0.99844479004665632000,
       0.99844720496894412000, 0.99844961240310082000, 0.99845201238390091000, 0.99845440494590421000,
       0.99845679012345678000, 0.99845916795069334000, 0.99846153846153851000, 0.99846390168970811000,
       0.99846625766871167000, 0.99846860643185298000, 0.99847094801223246000, 0.99847328244274813000,
       0.99847560975609762000, 0.99847792998477924000, 0.99848024316109418000, 0.99848254931714719000,
       0.99848484848484853000, 0.99848714069591527000, 0.99848942598187307000, 0.99849170437405732000,
       0.99849397590361444000, 0.99849624060150377000, 0.99849849849849848000, 0.99850074962518742000,
       0.99850299401197606000, 0.99850523168908822000, 0.99850746268656720000, 0.99850968703427723000,
       0.99851190476190477000, 0.99851411589895989000, 0.99851632047477745000, 0.99851851851851847000,
       0.99852071005917165000, 0.99852289512555392000, 0.99852507374631272000, 0.99852724594992637000,
       0.99852941176470589000, 0.99853157121879588000, 0.99853372434017595000, 0.99853587115666176000,
       0.99853801169590639000, 0.99854014598540142000, 0.99854227405247808000, 0.99854439592430855000,
       0.99854651162790697000, 0.99854862119013066000, 0.99855072463768113000, 0.99855282199710560000,
       0.99855491329479773000, 0.99855699855699853000, 0.99855907780979825000, 0.99856115107913668000,
       0.99856321839080464000, 0.99856527977044474000, 0.99856733524355301000, 0.99856938483547930000,
       0.99857142857142855000, 0.99857346647646217000, 0.99857549857549854000, 0.99857752489331442000,
       0.99857954545454541000, 0.99858156028368794000, 0.99858356940509918000, 0.99858557284299854000,
       0.99858757062146897000, 0.99858956276445698000, 0.99859154929577465000, 0.99859353023909991000,
       0.99859550561797750000, 0.99859747545582045000, 0.99859943977591037000, 0.99860139860139863000,
       0.99860335195530725000, 0.99860529986053004000, 0.99860724233983289000, 0.99860917941585536000,
       0.99861111111111112000, 0.99861303744798890000, 0.99861495844875348000, 0.99861687413554634000,
       0.99861878453038677000, 0.99862068965517237000, 0.99862258953168048000, 0.99862448418156813000,
       0.99862637362637363000, 0.99862825788751719000, 0.99863013698630132000, 0.99863201094391241000,
       0.99863387978142082000, 0.99863574351978168000, 0.99863760217983655000, 0.99863945578231295000,
       0.99864130434782605000, 0.99864314789687925000, 0.99864498644986455000, 0.99864682002706362000,
       0.99864864864864866000, 0.99865047233468285000, 0.99865229110512133000, 0.99865410497981155000,
       0.99865591397849462000, 0.99865771812080539000, 0.99865951742627346000, 0.99866131191432395000,
       0.99866310160427807000, 0.99866488651535379000, 0.99866666666666670000, 0.99866844207723038000,
       0.99867021276595747000, 0.99867197875166003000, 0.99867374005305043000, 0.99867549668874167000,
       0.99867724867724872000, 0.99867899603698806000, 0.99868073878627972000, 0.99868247694334655000,
       0.99868421052631584000, 0.99868593955321949000, 0.99868766404199472000, 0.99868938401048490000,
       0.99869109947643975000, 0.99869281045751634000, 0.99869451697127942000, 0.99869621903520212000,
       0.99869791666666663000, 0.99869960988296491000, 0.99870129870129876000, 0.99870298313878081000,
       0.99870466321243523000, 0.99870633893919791000, 0.99870801033591727000, 0.99870967741935479000,
       0.99871134020618557000, 0.99871299871299868000, 0.99871465295629824000, 0.99871630295250324000,
       0.99871794871794872000, 0.99871959026888601000, 0.99872122762148341000, 0.99872286079182626000,
       0.99872448979591832000, 0.99872611464968153000, 0.99872773536895676000, 0.99872935196950441000,
       0.99873096446700504000, 0.99873257287705952000, 0.99873417721518987000, 0.99873577749683939000,
       0.99873737373737370000, 0.99873896595208067000, 0.99874055415617125000, 0.99874213836477987000,
       0.99874371859296485000, 0.99874529485570895000, 0.99874686716791983000, 0.99874843554443049000,
       0.99875000000000003000, 0.99875156054931336000, 0.99875311720698257000, 0.99875466998754669000,
       0.99875621890547261000, 0.99875776397515525000, 0.99875930521091816000, 0.99876084262701359000,
       0.99876237623762376000, 0.99876390605686027000, 0.99876543209876545000, 0.99876695437731200000,
       0.99876847290640391000, 0.99876998769987702000, 0.99877149877149873000, 0.99877300613496933000,
       0.99877450980392157000, 0.99877600979192172000, 0.99877750611246940000, 0.99877899877899878000,
       0.99878048780487805000, 0.99878197320341044000, 0.99878345498783450000, 0.99878493317132444000,
       0.99878640776699024000, 0.99878787878787878000, 0.99878934624697335000, 0.99879081015719473000,
       0.99879227053140096000, 0.99879372738238847000, 0.99879518072289153000, 0.99879663056558365000,
       0.99879807692307687000, 0.99879951980792314000, 0.99880095923261392000, 0.99880239520958081000,
       0.99880382775119614000, 0.99880525686977295000, 0.99880668257756566000, 0.99880810488676997000,
       0.99880952380952381000, 0.99881093935790721000, 0.99881235154394299000, 0.99881376037959668000,
       0.99881516587677721000, 0.99881656804733732000, 0.99881796690307334000, 0.99881936245572611000,
       0.99882075471698117000, 0.99882214369846878000, 0.99882352941176467000, 0.99882491186839018000,
       0.99882629107981225000, 0.99882766705744430000, 0.99882903981264637000, 0.99883040935672518000,
       0.99883177570093462000, 0.99883313885647607000, 0.99883449883449882000, 0.99883585564610011000,
       0.99883720930232556000, 0.99883855981416958000, 0.99883990719257543000, 0.99884125144843572000,
       0.99884259259259256000, 0.99884393063583810000, 0.99884526558891451000, 0.99884659746251436000,
       0.99884792626728114000, 0.99884925201380903000, 0.99885057471264371000, 0.99885189437428246000,
       0.99885321100917435000, 0.99885452462772051000, 0.99885583524027455000, 0.99885714285714289000,
       0.99885844748858443000, 0.99885974914481190000, 0.99886104783599083000, 0.99886234357224113000,
       0.99886363636363640000, 0.99886492622020429000, 0.99886621315192747000, 0.99886749716874290000,
       0.99886877828054299000, 0.99887005649717520000, 0.99887133182844245000, 0.99887260428410374000,
       0.99887387387387383000, 0.99887514060742411000, 0.99887640449438198000, 0.99887766554433222000,
       0.99887892376681620000, 0.99888017917133254000, 0.99888143176733779000, 0.99888268156424576000,
       0.99888392857142860000, 0.99888517279821631000, 0.99888641425389757000, 0.99888765294771964000,
       0.99888888888888894000, 0.99889012208657046000, 0.99889135254988914000, 0.99889258028792915000,
       0.99889380530973448000, 0.99889502762430937000, 0.99889624724061810000, 0.99889746416758540000,
       0.99889867841409696000, 0.99889988998899892000, 0.99890109890109891000, 0.99890230515916578000,
       0.99890350877192979000, 0.99890470974808321000, 0.99890590809628010000, 0.99890710382513659000,
       0.99890829694323147000, 0.99890948745910579000, 0.99891067538126366000, 0.99891186071817195000,
       0.99891304347826082000, 0.99891422366992400000, 0.99891540130151846000, 0.99891657638136511000,
       0.99891774891774887000, 0.99891891891891893000, 0.99892008639308860000, 0.99892125134843579000,
       0.99892241379310343000, 0.99892357373519913000, 0.99892473118279568000, 0.99892588614393130000,
       0.99892703862660948000, 0.99892818863879962000, 0.99892933618843682000, 0.99893048128342243000,
       0.99893162393162394000, 0.99893276414087517000, 0.99893390191897657000, 0.99893503727369537000,
       0.99893617021276593000, 0.99893730074388953000, 0.99893842887473461000, 0.99893955461293749000,
       0.99894067796610164000, 0.99894179894179891000, 0.99894291754756870000, 0.99894403379091867000,
       0.99894514767932485000, 0.99894625922023184000, 0.99894736842105258000, 0.99894847528916930000,
       0.99894957983193278000, 0.99895068205666315000, 0.99895178197064993000, 0.99895287958115186000,
       0.99895397489539750000, 0.99895506792058519000, 0.99895615866388310000, 0.99895724713242962000,
       0.99895833333333328000, 0.99895941727367321000, 0.99896049896049899000, 0.99896157840083077000,
       0.99896265560165975000, 0.99896373056994814000, 0.99896480331262938000, 0.99896587383660806000,
       0.99896694214876036000, 0.99896800825593390000, 0.99896907216494846000, 0.99897013388259526000,
       0.99897119341563789000, 0.99897225077081198000, 0.99897330595482547000, 0.99897435897435893000,
       0.99897540983606559000, 0.99897645854657113000, 0.99897750511247441000, 0.99897854954034726000,
       0.99897959183673468000, 0.99898063200815490000, 0.99898167006109984000, 0.99898270600203454000,
       0.99898373983739841000, 0.99898477157360410000, 0.99898580121703850000, 0.99898682877406286000,
       0.99898785425101211000, 0.99898887765419619000, 0.99898989898989898000, 0.99899091826437947000,
       0.99899193548387100000, 0.99899295065458205000, 0.99899396378269623000, 0.99899497487437183000,
       0.99899598393574296000, 0.99899699097291872000, 0.99899799599198402000, 0.99899899899899902000,
       0.99900000000000000000, 0.99900099900099903000, 0.99900199600798401000, 0.99900299102691925000,
       0.99900398406374502000, 0.99900497512437814000, 0.99900596421471177000, 0.99900695134061568000,
       0.99900793650793651000, 0.99900891972249750000, 0.99900990099009901000, 0.99901088031651830000,
       0.99901185770750989000, 0.99901283316880551000, 0.99901380670611439000, 0.99901477832512320000,
       0.99901574803149606000, 0.99901671583087515000, 0.99901768172888017000, 0.99901864573110888000,
       0.99901960784313726000, 0.99902056807051909000, 0.99902152641878672000, 0.99902248289345064000};

void gauss_legendre_tbl(unsigned long n, double *x, double *w, double eps)
{
	double x0, x1, dx;	/* Abscissas */
	double w0 = 0, w1, dw;    /* Weights */
	double P0, P_1, P_2;      /* Legendre polynomial values */
	double dpdx;		  /* Legendre polynomial derivative */
	unsigned long i, j, k, m; /* Iterators */
	double t0, t1, t2, t3;

	m = (n + 1) >> 1;

	t0 = (1.0 - (1.0 - 1.0 / (double)n) / (8.0 * (double)n * (double)n));
	t1 = 1.0 / (4.0 * (double)n + 2.0);

	for (i = 1; i <= m; i++) {
		/* Find i-th root of Legendre polynomial */

		/* Initial guess */
		x0 = cos(PI * (double)((i << 2) - 1) * t1) * t0;

		/* Newton iterations, at least one */
		j = 0;
		dx = dw = DBL_MAX;
		do {
			/* Compute Legendre polynomial value at x0 */
			P_1 = 1.0;
			P0 = x0;
#if 0
			/* Simple, not optimized version */
			for (k = 2; k <= n; k++)
			{
				P_2 = P_1;
				P_1 = P0;
				t2 = x0*P_1;
				t3 = (double)(k-1)/(double)k;

				P0 = t2 + t3*(t2 - P_2);
			}
#else
			/* Optimized version using lookup tables */
			if (n < 1024) {
				/* Use fast algorithm for small n*/
				for (k = 2; k <= n; k++) {
					P_2 = P_1;
					P_1 = P0;
					t2 = x0 * P_1;

					P0 = t2 + ltbl[k] * (t2 - P_2);
				}
			} else {

				/* Use general algorithm for other n */
				for (k = 2; k < 1024; k++) {
					P_2 = P_1;
					P_1 = P0;
					t2 = x0 * P_1;

					P0 = t2 + ltbl[k] * (t2 - P_2);
				}

				for (k = 1024; k <= n; k++) {
					P_2 = P_1;
					P_1 = P0;
					t2 = x0 * P_1;
					t3 = (double)(k - 1) / (double)k;

					P0 = t2 + t3 * (t2 - P_2);
				}
			}
#endif
			/* Compute Legendre polynomial derivative at x0 */
			dpdx = ((x0 * P0 - P_1) * (double)n) / (x0 * x0 - 1.0);

			/* Newton step */
			x1 = x0 - P0 / dpdx;

			/* Weight computing */
			w1 = 2.0 / ((1.0 - x1 * x1) * dpdx * dpdx);

			/* Compute weight w0 on first iteration, needed for dw */
			if (j == 0)
				w0 = 2.0 / ((1.0 - x0 * x0) * dpdx * dpdx);

			dx = x0 - x1;
			dw = w0 - w1;

			x0 = x1;
			w0 = w1;
			j++;
		} while ((FABS(dx) > eps || FABS(dw) > eps) && j < 100);

		x[(m - 1) - (i - 1)] = x1;
		w[(m - 1) - (i - 1)] = w1;
	}

	return;
}

int gauss_grid_create(unsigned long size, double *x, double *w, double xmin, double xmax)
{
	double *xi, *wi;
	unsigned long j = 0, k = 0, i, sn;
	long int l;

	xi = malloc(size * sizeof(double));
	wi = malloc(size * sizeof(double));

	gauss_legendre_tbl(size, xi, wi, 1e-15);

	/* Sort in ascending order */

	for (l = (size % 2) ? (size / 2) : (size / 2 - 1); l >= 0; l--) {
		x[j++] = -xi[l];
		w[k++] = wi[l];
	}

	sn = (size % 2) ? (size / 2 + 1) : (size / 2);

	for (i = (size % 2) ? 1 : 0; i < sn; i++) {
		x[j++] = xi[i];
		w[k++] = wi[i];
	}

	for (i = 0; i < size; i++) {
		x[i] = 0.5 * (xmax + xmin) + 0.5 * (xmax - xmin) * x[i];
		w[i] *= 0.5 * (xmax - xmin);
	}

	free(xi);
	free(wi);

	return (0);
}

int gauss_grid_rescale(const double *x1, const double *w1, unsigned long size, double *x, double *w,
		       double xmin, double xmax)
{
	unsigned long i;

	for (i = 0; i < size; i++) {
		x[i] = 0.5 * (xmax + xmin) + 0.5 * (xmax - xmin) * x1[i];
		w[i] = 0.5 * (xmax - xmin) * w1[i];
	}

	return 0;
}

double test_gauss_grid_create(unsigned long n, int tfun)
{
	unsigned long i;
	double *x, *w;
	double a = 0, b = 2, In[3], Incomp[3], Indiff[3];

	fprintf(stderr, "test_gauss_grid_create() test #%d %s:%d\n", tfun, __FILE__, __LINE__);

	x = malloc(n * sizeof(double));
	assert(x);
	w = malloc(n * sizeof(double));
	assert(w);

	gauss_grid_create(n, x, w, a, b);

	In[0] = 0;
	In[1] = 0;
	In[2] = 0;
	for (i = 0; i < n; i++) {
		In[0] += x[i] * x[i] * w[i];
		In[1] += sin(x[i]) * w[i];
		In[2] += exp(-x[i]) * w[i];
	}

	Incomp[0] = (b * b * b / 3) - (a * a * a / 3);
	Incomp[1] = cos(a) - cos(b);
	Incomp[2] = exp(-a) - exp(-b);

	Indiff[0] = fabs(Incomp[0] - In[0]);
	Indiff[1] = fabs(Incomp[1] - In[1]);
	Indiff[2] = fabs(Incomp[2] - In[2]);

	free(x);
	free(w);

	return Indiff[tfun];
}
