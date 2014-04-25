function res=formatString(value)

value=abs(value);
res=' %5.8f ';

if(le(value,100000)&&ge(value,10000))
    res=[' ' res ''];
elseif(le(value,10000)&&ge(value,1000))
    res=[' ' res ' '];
    
elseif(le(value,1000)&&ge(value,100))
    res=[' ' res '  '];
    
elseif (le(value,100)&&ge(value,10))
    res=[' ' res '   '];
    
elseif (le(value,10))
    res=[' ' res '    '];
    
end