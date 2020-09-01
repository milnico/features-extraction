
trial=(1 2 3 4 5 );


      for t in "${trial[@]}"
      do
         nohup python es.py -f configuration.ini -s $t 2>&1 > seedS$t.out &
      done

