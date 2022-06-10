def print_svg(src, tgt, aggr_src, aggr_tgt, align):
        start_x = 25
        start_y = 100
        len_square = 15
        len_x = len(tgt)
        len_y = len(src)
        separation = 2
        print "<br>\n<svg width=\""+str(len_x*len_square + start_x + 150)+"\" height=\""+str(len_y*len_square + start_y)+"\">"
        for x in range(len(tgt)): ### tgt
            if aggr_tgt[x]<0: col="red"
            else: col="black"
            print "<text x=\""+str(x*len_square + start_x)+"\" y=\""+str(start_y-2)+"\" fill=\""+col+"\" font-family=\"Courier\" font-size=\"5\">"+"{:+.1f}".format(aggr_tgt[x])+"</text>"
            col="black" ### remove this line if you want divergent words in red
            print "<text x=\""+str(x*len_square + start_x + separation)+"\" y=\""+str(start_y-15)+"\" fill=\""+col+"\" font-family=\"Courier\" font-size=\"10\" transform=\"rotate(-45 "+str(x*len_square + start_x + 10)+","+str(start_y-15)+") \">"+tgt[x]+"</text>"
        for y in range(len(src)): ### src
            for x in range(len(tgt)): ### tgt
                color = align[y][x]
                if color < 0: color = 1
                elif color > 10: color = 0
                else: color = (-color+10)/10
                color = int(color*256)
                print "<rect x=\""+str(x*len_square + start_x)+"\" y=\""+str(y*len_square + start_y)+"\" width=\""+str(len_square)+"\" height=\""+str(len_square)+"\" style=\"fill:rgb("+str(color)+","+str(color)+","+str(color)+"); stroke-width:1;stroke:rgb(200,200,200)\" />"
                txtcolor = "black"
                if align[y][x] < 0: txtcolor="red"
                print "<text x=\""+str(x*len_square + start_x)+"\" y=\""+str(y*len_square + start_y + len_square*3/4)+"\" fill=\"{}\" font-family=\"Courier\" font-size=\"5\">".format(txtcolor)+"{:+.1f}".format(align[y][x])+"</text>"

            if aggr_src[y]<0: col="red" ### last column with source words
            else: col="black"
            print "<text x=\""+str(len_x*len_square + start_x + separation)+"\" y=\""+str(y*len_square + start_y + len_square*3/4)+"\" fill=\""+col+"\" font-family=\"Courier\" font-size=\"5\">"+"{:+.1f}".format(aggr_src[y])+"</text>"
            col="black" ### remove this line if you want divergent words in red
            print "<text x=\""+str(len_x*len_square + start_x + separation + 15)+"\" y=\""+str(y*len_square + start_y + len_square*3/4)+"\" fill=\""+col+"\" font-family=\"Courier\" font-size=\"10\">"+src[y]+"</text>"
        print("<br>\n<svg width=\"200\" height=\"20\">")
        
