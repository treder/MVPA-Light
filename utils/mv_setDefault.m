function isdefault= mv_setDefault(opt,tag,value)
% Set default value for a field in a struct. If opt.tag does not exist, it
% is set to opt.tag = value

  stridx= strfind(tag,'.'); % check whether a substruct is referenced
  if ~isempty(stridx)
      tagstart= ['.' tag(1:stridx(end)-1)];
      tagend= tag(stridx(end)+1:end);
  else
      tagstart= [];
      tagend= tag;
  end
  if ~evalin('caller',['isfield('  inputname(1) tagstart ',''' tagend ''')'])
    eval(['opt.' tag ' = value;'])
    % Transfer new value to calling function
    assignin('caller',inputname(1),opt);    
    isdefault= 1;
  else
    isdefault= 0;
  end
end

