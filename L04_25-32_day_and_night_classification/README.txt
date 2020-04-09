Een simpele image classifier die dag en nachtfotos van elkaar kan scheiden.

Het project maakt gebruik van de volgende pipeline aan stappen:

visualisatie (en ook tussendoor) ==> pre-processing ==> feature extraction ==> classificatie
                                     [ fotos zelfde     [ helderheid           [ is de foto
                                      dimensie            uit de foto            boven de
                                      maken &             halen           ]      threshold? ]
                                      veranderen
                                      van kleur-
                                      model       ]
De classificatie werkt vrij simpel: een gemiddelde waarde voor de helderheid van een foto (V in HSV) wordt berekend.
De classifier kijkt of de gem. helderheid van de foto lager is dan de threshold. Als deze lager is, is het een
nachtfoto; anders een dagfoto.